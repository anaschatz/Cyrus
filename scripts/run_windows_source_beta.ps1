[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $HandoffManifest,
    [Parameter(Mandatory = $true)] [string] $StackConfig,
    [ValidateSet("CpuBaseline", "CudaTurnScoped")]
    [string] $Profile = "CpuBaseline",
    [ValidateRange(15, 600)] [int] $StartupTimeoutSeconds = 180,
    [switch] $PreflightOnly,
    [switch] $RunE2E,
    [switch] $StopAfterCheck
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "windows_source_common.ps1")
$OriginalProcessEnvironment = Get-CyrusProcessEnvironmentSnapshot
$OwnedProcesses = [System.Collections.Generic.List[object]]::new()
$SourceProcessJob = $null
$PrivateRuntimeDirectory = $null
try {
$Repository = Get-CyrusRepositoryRoot
$StackConfigPath = Resolve-CyrusRequiredFile $StackConfig "windows_source.stack_config_missing"
$PrivateRuntimeParent = Assert-CyrusNoReparsePath `
    -LiteralPath (Split-Path -Parent $StackConfigPath) `
    -RequireContainer
Assert-CyrusPathOutsideRepository `
    -LiteralPath $PrivateRuntimeParent `
    -Repository $Repository
$RuntimeItem = New-Item `
    -ItemType Directory `
    -Path (Join-Path `
        $PrivateRuntimeParent `
        ("cyrus-source-runtime-" + [Guid]::NewGuid().ToString("N"))
    ) `
    -ErrorAction Stop
$PrivateRuntimeDirectory = Assert-CyrusNoReparsePath `
    -LiteralPath $RuntimeItem.FullName `
    -RequireContainer

# Resolve tool identities without executing them, then replace the complete
# parent environment before any external validator/tool is launched.
$ResolvedTools = @{}
foreach ($Name in @("py", "uv", "git")) {
    $Command = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue
    if ($null -eq $Command) {
        throw "windows_source.required_tool_missing"
    }
    $ResolvedTools[$Name] = Assert-CyrusNoReparsePath `
        -LiteralPath $Command.Source `
        -RequireLeaf
}
$ToolDirectories = @(
    $ResolvedTools.Values |
        ForEach-Object { Split-Path -Parent $_ } |
        Sort-Object -Unique
)
$ValidatorEnvironment = New-CyrusSourceChildEnvironment `
    -Role Validator `
    -PrivateRuntimeDirectory $PrivateRuntimeDirectory `
    -AdditionalPathDirectories $ToolDirectories
Set-CyrusProcessEnvironmentEntries -EnvironmentEntries $ValidatorEnvironment
foreach ($Name in @(
    "JARVIS_GATEWAY_WHEEL",
    "JARVIS_QUAL_GPU_DEVICE",
    "JARVIS_QUAL_GPU_MEMORY_BYTES",
    "JARVIS_QUAL_GPU_DRIVER",
    "JARVIS_QUAL_COMPUTE_RUNTIME",
    "JARVIS_QUAL_COMPUTE_CAPABILITY"
)) {
    if ($OriginalProcessEnvironment.ContainsKey($Name)) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            $OriginalProcessEnvironment[$Name],
            [EnvironmentVariableTarget]::Process
        )
    }
}
$Toolchain = Assert-CyrusNativeWindowsToolchain `
    -PythonLauncherPath $ResolvedTools["py"] `
    -UvExecutablePath $ResolvedTools["uv"]
Set-CyrusRuntimeOfflineEnvironment

$StackDocument = Get-Content -LiteralPath $StackConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$VoiceConfigPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.voice_config_path) `
    "windows_source.voice_config_missing"
$OrchestratorConfigPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.orchestrator_config_path) `
    "windows_source.orchestrator_config_missing"
$InferenceManifestPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.inference.manifest_path) `
    "windows_source.inference_manifest_missing"
$OrchestratorDocument = Get-Content `
    -LiteralPath $OrchestratorConfigPath `
    -Raw `
    -Encoding UTF8 | ConvertFrom-Json
$ConfiguredInferenceBearer = [string]$OrchestratorDocument.inference_bearer_token
if (
    [string]::IsNullOrWhiteSpace($ConfiguredInferenceBearer) -or
    $ConfiguredInferenceBearer.Length -gt 4096 -or
    [Text.Encoding]::UTF8.GetByteCount($ConfiguredInferenceBearer) -lt 32
) {
    throw "windows_source.inference_auth_invalid"
}
$VoicePython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository ".venv\Scripts\python.exe") `
    "windows_source.voice_environment_missing"
$InferencePython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository "inference\.venv\Scripts\python.exe") `
    "windows_source.inference_environment_missing"
$OrchestratorPython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository "orchestrator\.venv\Scripts\python.exe") `
    "windows_source.orchestrator_environment_missing"

function Wait-CyrusHttpReady {
    param(
        [Parameter(Mandatory = $true)] [object] $Process,
        [Parameter(Mandatory = $true)] [DateTime] $Deadline,
        [Parameter(Mandatory = $true)] [string] $Uri,
        [Parameter(Mandatory = $true)] [string] $HostHeader,
        [Parameter(Mandatory = $true)] [string] $ExitFailure,
        [Parameter(Mandatory = $true)] [string] $TimeoutFailure
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($Process.HasExited) {
            throw $ExitFailure
        }
        try {
            $Response = Invoke-RestMethod `
                -Method Get `
                -Uri $Uri `
                -Headers @{ Host = $HostHeader } `
                -TimeoutSec 2
            if ($Response.ready -eq $true) {
                return
            }
        }
        catch {
            # A bounded retry is expected while a local component starts.
        }
        Start-Sleep -Milliseconds 250
    }
    throw $TimeoutFailure
}

function Wait-CyrusStackReady {
    param(
        [Parameter(Mandatory = $true)] [object[]] $Processes,
        [Parameter(Mandatory = $true)] [DateTime] $Deadline
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        foreach ($Process in $Processes) {
            if ($Process.HasExited) {
                throw "windows_source.component_exited"
            }
        }
        $StatusRaw = & $OrchestratorPython `
            -I -m jarvis_orchestrator.stack_cli `
            --config $StackConfigPath `
            status 2>$null
        if ($LASTEXITCODE -eq 0) {
            try {
                $Status = $StatusRaw | ConvertFrom-Json
                if (
                    $Status.ready -eq $true -and
                    $Status.services_ready -eq $true -and
                    $Status.voice_client_connected -eq $true
                ) {
                    return
                }
            }
            catch {
                # The next bounded poll performs a fresh closed status check.
            }
        }
        Start-Sleep -Milliseconds 250
    }
    throw "windows_source.stack_start_timeout"
}

try {
    # Preflight runs with no bearer in the process environment.  Its model,
    # config and qualification checks do not need the serving credential.
    & (Join-Path $PSScriptRoot "preflight_windows_source.ps1") `
        -HandoffManifest $HandoffManifest `
        -StackConfig $StackConfigPath `
        -Profile $Profile

    if ($PreflightOnly) {
        [PSCustomObject]@{
            schema = "cyrus.windows-source-supervised-preflight/1"
            ready = $true
            source_run_only = $true
            production_ready = $false
            inference_bearer_not_exported = $true
        } | ConvertTo-Json -Compress
        return
    }

    $Deadline = [DateTime]::UtcNow.AddSeconds($StartupTimeoutSeconds)
    $SourceProcessJob = New-CyrusSourceProcessJob
    $InferenceEnvironment = New-CyrusSourceChildEnvironment `
        -Role Inference `
        -PrivateRuntimeDirectory $PrivateRuntimeDirectory `
        -InferenceBearer $ConfiguredInferenceBearer
    $OrchestratorEnvironment = New-CyrusSourceChildEnvironment `
        -Role Orchestrator `
        -PrivateRuntimeDirectory $PrivateRuntimeDirectory
    $VoiceEnvironment = New-CyrusSourceChildEnvironment `
        -Role Voice `
        -PrivateRuntimeDirectory $PrivateRuntimeDirectory

    $InferenceProcess = Start-CyrusSourceJobProcess `
        -Job $SourceProcessJob `
        -Executable $InferencePython `
        -Arguments @(
            "-I", "-m", "jarvis_inference", "serve", "--manifest", $InferenceManifestPath
        ) `
        -WorkingDirectory $Repository `
        -EnvironmentEntries $InferenceEnvironment
    $OwnedProcesses.Add($InferenceProcess)
    Wait-CyrusHttpReady `
        -Process $InferenceProcess `
        -Deadline $Deadline `
        -Uri "http://127.0.0.1:8766/readyz" `
        -HostHeader "127.0.0.1:8766" `
        -ExitFailure "windows_source.inference_exited" `
        -TimeoutFailure "windows_source.inference_start_timeout"

    $OrchestratorProcess = Start-CyrusSourceJobProcess `
        -Job $SourceProcessJob `
        -Executable $OrchestratorPython `
        -Arguments @(
            "-I", "-m", "jarvis_orchestrator", "--config", $OrchestratorConfigPath, "serve"
        ) `
        -WorkingDirectory $Repository `
        -EnvironmentEntries $OrchestratorEnvironment
    $OwnedProcesses.Add($OrchestratorProcess)
    Wait-CyrusHttpReady `
        -Process $OrchestratorProcess `
        -Deadline $Deadline `
        -Uri "http://127.0.0.1:8767/readyz" `
        -HostHeader "127.0.0.1:8767" `
        -ExitFailure "windows_source.orchestrator_exited" `
        -TimeoutFailure "windows_source.orchestrator_start_timeout"

    $VoiceProcess = Start-CyrusSourceJobProcess `
        -Job $SourceProcessJob `
        -Executable $VoicePython `
        -Arguments @(
            "-I", "-m", "jarvis_voice", "run", "--config", $VoiceConfigPath
        ) `
        -WorkingDirectory $Repository `
        -EnvironmentEntries $VoiceEnvironment
    $OwnedProcesses.Add($VoiceProcess)
    Wait-CyrusStackReady -Processes $OwnedProcesses.ToArray() -Deadline $Deadline

    if ($RunE2E) {
        Invoke-CyrusCheckedCommand $OrchestratorPython @(
            "-I", "-m", "jarvis_orchestrator.stack_cli"
            "--config", $StackConfigPath
            "e2e", "--turns", "1"
        ) "windows_source.e2e_failed"
    }

    [PSCustomObject]@{
        schema = "cyrus.windows-source-beta-session/1"
        ready = $true
        source_run_only = $true
        production_ready = $false
        native_qualification_complete = $false
        profile = $Profile
        full_local_stack_connected = $true
        e2e_exercised = [bool]$RunE2E
        autostart_installed = $false
    } | ConvertTo-Json -Compress

    if (-not $StopAfterCheck) {
        Write-Host "CYRUS source-run beta is active. Press Ctrl+C to stop all three owned processes."
        while ($true) {
            foreach ($Process in $OwnedProcesses) {
                if ($Process.HasExited) {
                    throw "windows_source.component_exited"
                }
            }
            Start-Sleep -Milliseconds 500
        }
    }
}
finally {
    $JobCloseFailed = $false
    if ($null -ne $SourceProcessJob) {
        try {
            # Closing the Job Object is the authoritative descendant cleanup.
            # It happens before checking the recorded top-level parents, so an
            # already-exited parent cannot hide a still-running child.
            $SourceProcessJob.Dispose()
        }
        catch {
            $JobCloseFailed = $true
            try {
                # If closing the job handle itself failed, terminate the
                # complete tree through that same authoritative job handle.
                $SourceProcessJob.TerminateAll()
            }
            catch {
                # Survivor checks below remain authoritative.
            }
        }
    }
    for ($Index = $OwnedProcesses.Count - 1; $Index -ge 0; $Index--) {
        Stop-CyrusOwnedProcess $OwnedProcesses[$Index]
    }
    if ($JobCloseFailed -and $null -ne $SourceProcessJob) {
        try {
            $SourceProcessJob.Dispose()
            $JobCloseFailed = $false
        }
        catch {
            # The fixed failure code below remains authoritative.
        }
    }
    $OwnedSurvivor = $false
    foreach ($Process in $OwnedProcesses) {
        if (-not $Process.HasExited) {
            $OwnedSurvivor = $true
        }
    }
    $HandleCloseFailed = $false
    foreach ($Process in $OwnedProcesses) {
        try {
            $Process.Dispose()
        }
        catch {
            $HandleCloseFailed = $true
        }
    }
    $RuntimeCleanupFailed = $false
    if (
        -not $OwnedSurvivor -and
        -not $JobCloseFailed -and
        -not $HandleCloseFailed -and
        $null -ne $PrivateRuntimeDirectory
    ) {
        try {
            $CheckedRuntimeDirectory = Assert-CyrusNoReparsePath `
                -LiteralPath $PrivateRuntimeDirectory `
                -RequireContainer
            Remove-Item `
                -LiteralPath $CheckedRuntimeDirectory `
                -Recurse `
                -Force `
                -ErrorAction Stop
            $PrivateRuntimeDirectory = $null
        }
        catch {
            $RuntimeCleanupFailed = $true
        }
    }
    if ($OwnedSurvivor) {
        throw "windows_source.orphan_process_detected"
    }
    if ($JobCloseFailed) {
        throw "windows_source.job_close_failed"
    }
    if ($HandleCloseFailed) {
        throw "windows_source.process_handle_close_failed"
    }
    if ($RuntimeCleanupFailed) {
        throw "windows_source.runtime_directory_cleanup_failed"
    }
}
}
finally {
    if (
        $null -ne $PrivateRuntimeDirectory -and
        $null -eq $SourceProcessJob -and
        $OwnedProcesses.Count -eq 0
    ) {
        try {
            $CheckedRuntimeDirectory = Assert-CyrusNoReparsePath `
                -LiteralPath $PrivateRuntimeDirectory `
                -RequireContainer
            Remove-Item `
                -LiteralPath $CheckedRuntimeDirectory `
                -Recurse `
                -Force `
                -ErrorAction Stop
            $PrivateRuntimeDirectory = $null
        }
        catch {
            # The original setup failure remains authoritative.  A reparse
            # replacement is never traversed for cleanup.
        }
    }
    Restore-CyrusProcessEnvironmentSnapshot -Snapshot $OriginalProcessEnvironment
}
