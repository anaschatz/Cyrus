[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "windows_source_common.ps1")

if ($env:OS -ne "Windows_NT") {
    throw "windows_source.native_windows_required"
}

foreach ($RejectedPath in @(
    "relative\asset.bin",
    "C:drive-relative.bin",
    "\\server\share\asset.bin",
    "\\?\C:\asset.bin",
    "\\.\pipe\cyrus"
)) {
    if (Test-CyrusLocalDriveAbsolutePath $RejectedPath) {
        throw "windows_source.path_compatibility_false_accept"
    }
}
foreach ($AcceptedPath in @("C:\asset.bin", "Z:/asset.bin")) {
    if (-not (Test-CyrusLocalDriveAbsolutePath $AcceptedPath)) {
        throw "windows_source.path_compatibility_false_reject"
    }
}

$Temporary = Join-Path `
    ([IO.Path]::GetTempPath()) `
    ("cyrus-job-smoke-" + [Guid]::NewGuid().ToString("N"))
[void](New-Item -ItemType Directory -Path $Temporary)
$Temporary = Assert-CyrusNoReparsePath `
    -LiteralPath $Temporary `
    -RequireContainer
$PidPath = Join-Path $Temporary "child.pid"
$ParentScript = Assert-CyrusNoReparsePath `
    -LiteralPath (Join-Path $PSScriptRoot "windows_source_job_smoke_parent.ps1") `
    -RequireLeaf
$PowerShell = Assert-CyrusNoReparsePath `
    -LiteralPath (Join-Path (
        [Environment]::SystemDirectory
    ) "WindowsPowerShell\v1.0\powershell.exe") `
    -RequireLeaf

$Job = $null
$Parent = $null
$Child = $null
$VoiceProbe = $null
$InferenceProbe = $null
$ValidatorProbe = $null
$OriginalEnvironment = Get-CyrusProcessEnvironmentSnapshot
try {
    foreach ($Name in @(
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "AZURE_CLIENT_SECRET",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PYTHONPATH",
        "PYTHONHOME",
        "JARVIS_INFERENCE_API_KEY"
    )) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            "must-not-be-inherited",
            [EnvironmentVariableTarget]::Process
        )
    }
    foreach ($Name in @(
        "JARVIS_GATEWAY_WHEEL",
        "JARVIS_QUAL_GPU_DEVICE",
        "JARVIS_QUAL_GPU_MEMORY_BYTES",
        "JARVIS_QUAL_GPU_DRIVER",
        "JARVIS_QUAL_COMPUTE_RUNTIME",
        "JARVIS_QUAL_COMPUTE_CAPABILITY"
    )) {
        [Environment]::SetEnvironmentVariable(
            $Name,
            "native-smoke-value",
            [EnvironmentVariableTarget]::Process
        )
    }

    $VoiceEnvironment = New-CyrusSourceChildEnvironment `
        -Role Voice `
        -PrivateRuntimeDirectory $Temporary
    $InferenceBearer = "native-smoke-bearer-value-1234567890"
    $InferenceEnvironment = New-CyrusSourceChildEnvironment `
        -Role Inference `
        -PrivateRuntimeDirectory $Temporary `
        -InferenceBearer $InferenceBearer
    $ValidatorEnvironment = New-CyrusSourceChildEnvironment `
        -Role Validator `
        -PrivateRuntimeDirectory $Temporary
    Set-CyrusProcessEnvironmentEntries -EnvironmentEntries $ValidatorEnvironment
    $Job = New-CyrusSourceProcessJob

    $VoiceProbeSource = @'
if ($null -ne [Environment]::GetEnvironmentVariable("JARVIS_INFERENCE_API_KEY")) { exit 31 }
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "AZURE_CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH", "PYTHONHOME")) {
    if ($null -ne [Environment]::GetEnvironmentVariable($name)) { exit 32 }
}
if ($env:CYRUS_UNICODE_PROBE -cne "Κύρος") { exit 33 }
if ($env:NO_PROXY -cne "127.0.0.1,localhost") { exit 34 }
'@
    $VoiceProbeEncoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($VoiceProbeSource)
    )
    $VoiceProbe = Start-CyrusSourceJobProcess `
        -Job $Job `
        -Executable $PowerShell `
        -Arguments @(
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            $VoiceProbeEncoded
        ) `
        -WorkingDirectory $Temporary `
        -EnvironmentEntries @($VoiceEnvironment + "CYRUS_UNICODE_PROBE=Κύρος")
    if (-not $VoiceProbe.WaitForExit(15000) -or $VoiceProbe.ExitCode -ne 0) {
        throw "windows_source.job_smoke_voice_environment_failed"
    }

    $ValidatorProbe = Start-CyrusSourceJobProcess `
        -Job $Job `
        -Executable $PowerShell `
        -Arguments @(
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            $VoiceProbeEncoded
        ) `
        -WorkingDirectory $Temporary `
        -EnvironmentEntries @($ValidatorEnvironment + "CYRUS_UNICODE_PROBE=Κύρος")
    if (-not $ValidatorProbe.WaitForExit(15000) -or $ValidatorProbe.ExitCode -ne 0) {
        throw "windows_source.job_smoke_validator_environment_failed"
    }

    $InferenceProbeSource = @'
if ($env:JARVIS_INFERENCE_API_KEY -cne "native-smoke-bearer-value-1234567890") { exit 41 }
foreach ($name in @("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "OPENAI_API_KEY", "AWS_SECRET_ACCESS_KEY", "AZURE_CLIENT_SECRET", "GOOGLE_APPLICATION_CREDENTIALS", "PYTHONPATH", "PYTHONHOME")) {
    if ($null -ne [Environment]::GetEnvironmentVariable($name)) { exit 42 }
}
'@
    $InferenceProbeEncoded = [Convert]::ToBase64String(
        [Text.Encoding]::Unicode.GetBytes($InferenceProbeSource)
    )
    $InferenceProbe = Start-CyrusSourceJobProcess `
        -Job $Job `
        -Executable $PowerShell `
        -Arguments @(
            "-NoProfile",
            "-NonInteractive",
            "-EncodedCommand",
            $InferenceProbeEncoded
        ) `
        -WorkingDirectory $Temporary `
        -EnvironmentEntries $InferenceEnvironment
    if (-not $InferenceProbe.WaitForExit(15000) -or $InferenceProbe.ExitCode -ne 0) {
        throw "windows_source.job_smoke_inference_environment_failed"
    }

    $Parent = Start-CyrusSourceJobProcess `
        -Job $Job `
        -Executable $PowerShell `
        -Arguments @(
            "-NoProfile",
            "-NonInteractive",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            $ParentScript,
            "-ChildPidPath",
            $PidPath
        ) `
        -WorkingDirectory $Temporary `
        -EnvironmentEntries $VoiceEnvironment
    $Stop = [DateTime]::UtcNow.AddSeconds(15)
    while (-not (Test-Path -LiteralPath $PidPath -PathType Leaf)) {
        if ($Parent.HasExited) {
            throw "windows_source.job_smoke_parent_failed"
        }
        if ([DateTime]::UtcNow -ge $Stop) {
            throw "windows_source.job_smoke_pid_missing"
        }
        Start-Sleep -Milliseconds 25
    }
    $RawPid = [IO.File]::ReadAllText($PidPath, [Text.Encoding]::ASCII)
    if ($RawPid -notmatch "^[1-9][0-9]{0,9}$") {
        throw "windows_source.job_smoke_pid_invalid"
    }
    $Child = [Diagnostics.Process]::GetProcessById([int]$RawPid)
    $Child.Refresh()
    if ($Child.HasExited) {
        throw "windows_source.job_smoke_child_exited_early"
    }

    $Parent.Terminate()
    if (-not $Parent.WaitForExit(5000)) {
        throw "windows_source.job_smoke_parent_timeout"
    }
    if ($Parent.ExitCode -ne 1) {
        throw "windows_source.job_smoke_parent_failed"
    }

    # The parent has exited through its authoritative handle. Closing the only
    # Job Object handle must still terminate its orphaned sleeper child.
    $Job.Dispose()
    $Job = $null
    if (-not $Child.WaitForExit(10000)) {
        throw "windows_source.job_smoke_orphan_survived"
    }
    [PSCustomObject]@{
        schema = "cyrus.windows-source-job-smoke/1"
        powershell_51_path_contract = $true
        explicit_unicode_environment = $true
        ambient_secret_environment_stripped = $true
        validator_environment_stripped = $true
        inference_bearer_scoped = $true
        authoritative_process_handle = $true
        parent_exited_before_job_close = $true
        orphan_child_killed_on_job_close = $true
    } | ConvertTo-Json -Compress
}
finally {
    if ($null -ne $Job) {
        try {
            $Job.Dispose()
        }
        catch {
            # The primary failure remains authoritative in this test process.
        }
    }
    if ($null -ne $Parent) {
        Stop-CyrusOwnedProcess $Parent
        $Parent.Dispose()
    }
    if ($null -ne $VoiceProbe) {
        Stop-CyrusOwnedProcess $VoiceProbe
        $VoiceProbe.Dispose()
    }
    if ($null -ne $InferenceProbe) {
        Stop-CyrusOwnedProcess $InferenceProbe
        $InferenceProbe.Dispose()
    }
    if ($null -ne $ValidatorProbe) {
        Stop-CyrusOwnedProcess $ValidatorProbe
        $ValidatorProbe.Dispose()
    }
    if ($null -ne $Child) {
        $Child.Dispose()
    }
    Restore-CyrusProcessEnvironmentSnapshot -Snapshot $OriginalEnvironment
    if (Test-Path -LiteralPath $Temporary -PathType Container) {
        Remove-Item -LiteralPath $Temporary -Recurse -Force
    }
}
