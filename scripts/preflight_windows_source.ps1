[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $HandoffManifest,
    [Parameter(Mandatory = $true)] [string] $StackConfig,
    [ValidateSet("CpuBaseline", "CudaTurnScoped")]
    [string] $Profile = "CpuBaseline"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "windows_source_common.ps1")
$OriginalProcessEnvironment = Get-CyrusProcessEnvironmentSnapshot
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
        ("cyrus-preflight-runtime-" + [Guid]::NewGuid().ToString("N"))
    ) `
    -ErrorAction Stop
$PrivateRuntimeDirectory = Assert-CyrusNoReparsePath `
    -LiteralPath $RuntimeItem.FullName `
    -RequireContainer

# Resolve but do not execute the approved tools while ambient PATH is still
# visible.  Every subprocess below sees only the explicit validator block.
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
Assert-CyrusTrackedTreeClean `
    -Repository $Repository `
    -GitExecutable $ResolvedTools["git"]
$FrozenCommit = Get-CyrusFrozenCommit `
    -Repository $Repository `
    -GitExecutable $ResolvedTools["git"]
Assert-CyrusHandoffManifest `
    -Repository $Repository `
    -PythonExecutable $Toolchain.PythonExecutable `
    -HandoffManifest $HandoffManifest `
    -ExpectedCommit $FrozenCommit
$VoicePython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository ".venv\Scripts\python.exe") `
    "windows_source.voice_environment_missing"
$InferencePython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository "inference\.venv\Scripts\python.exe") `
    "windows_source.inference_environment_missing"
$OrchestratorPython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository "orchestrator\.venv\Scripts\python.exe") `
    "windows_source.orchestrator_environment_missing"

Invoke-CyrusCheckedCommand $OrchestratorPython @(
    "-I", "-m", "jarvis_orchestrator.stack_cli"
    "--config", $StackConfigPath
    "validate"
) "windows_source.stack_config_invalid"

try {
    $StackDocument = Get-Content -LiteralPath $StackConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
}
catch {
    throw "windows_source.stack_config_unreadable"
}
$VoiceConfigPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.voice_config_path) `
    "windows_source.voice_config_missing"
$OrchestratorConfigPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.orchestrator_config_path) `
    "windows_source.orchestrator_config_missing"
$InferenceManifestPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.inference.manifest_path) `
    "windows_source.inference_manifest_missing"
$InferenceQualificationPath = Resolve-CyrusRequiredFile `
    ([string]$StackDocument.inference.qualification_report_path) `
    "windows_source.inference_qualification_missing"
foreach ($PrivatePath in @(
    $VoiceConfigPath,
    $OrchestratorConfigPath,
    $InferenceManifestPath,
    $InferenceQualificationPath
)) {
    Assert-CyrusPathOutsideRepository -LiteralPath $PrivatePath -Repository $Repository
}

try {
    $VoiceDocument = Get-Content -LiteralPath $VoiceConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $OrchestratorDocument = Get-Content -LiteralPath $OrchestratorConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
}
catch {
    throw "windows_source.component_config_unreadable"
}
$ConfiguredInferenceBearer = [string]$OrchestratorDocument.inference_bearer_token
if (
    [string]::IsNullOrWhiteSpace($ConfiguredInferenceBearer) -or
    $ConfiguredInferenceBearer.Length -gt 4096 -or
    [Text.Encoding]::UTF8.GetByteCount($ConfiguredInferenceBearer) -lt 32 -or
    $ConfiguredInferenceBearer.Contains("`0")
) {
    throw "windows_source.inference_auth_invalid"
}
if (
    $VoiceDocument.orchestrator_url -ne "ws://127.0.0.1:8765" -or
    $OrchestratorDocument.voice_host -ne "127.0.0.1" -or
    $OrchestratorDocument.voice_port -ne 8765 -or
    $OrchestratorDocument.control_host -ne "127.0.0.1" -or
    $OrchestratorDocument.control_port -ne 8767 -or
    $OrchestratorDocument.inference_base_url -ne "http://127.0.0.1:8766/v1"
) {
    throw "windows_source.fixed_endpoints_invalid"
}
if (
    $VoiceDocument.windows_tts_backend -ne "sapi5" -or
    [string]::IsNullOrWhiteSpace([string]$VoiceDocument.sapi_voices.'el-GR') -or
    [string]::IsNullOrWhiteSpace([string]$VoiceDocument.sapi_voices.'en-US')
) {
    throw "windows_source.sapi_baseline_not_materialized"
}
if (
    $VoiceDocument.scheduler.enabled -ne $false -or
    $OrchestratorDocument.memory_enabled -ne $false -or
    $OrchestratorDocument.memory_shadow_enabled -ne $false -or
    $OrchestratorDocument.research_enabled -ne $false -or
    $OrchestratorDocument.cloud_enabled -ne $false -or
    $OrchestratorDocument.tools_enabled -ne $false
) {
    throw "windows_source.optional_capability_enabled"
}
if ($Profile -eq "CpuBaseline") {
    if (
        $VoiceDocument.asr.model_profile -ne "base" -or
        $VoiceDocument.asr.execution_profile -ne "cpu" -or
        $VoiceDocument.asr.cpu_compute_type -ne "int8"
    ) {
        throw "windows_source.cpu_baseline_profile_invalid"
    }
}
else {
    if (
        $VoiceDocument.asr.model_profile -ne "large-v3-turbo" -or
        $VoiceDocument.asr.execution_profile -ne "cuda" -or
        $VoiceDocument.asr.cpu_compute_type -ne "int8" -or
        $VoiceDocument.asr.cuda.device_index -ne 0 -or
        $VoiceDocument.asr.cuda.compute_type -ne "int8_float16" -or
        $VoiceDocument.asr.cuda.residency -ne "turn_scoped" -or
        $VoiceDocument.asr.cuda.minimum_free_vram_mb -ne 2048 -or
        $VoiceDocument.asr.cuda.reserve_for_other_workloads_mb -ne 2048 -or
        $VoiceDocument.asr.cuda.fallback_to_cpu -ne $true
    ) {
        throw "windows_source.cuda_turn_scoped_profile_invalid"
    }
}

if (
    [string]::IsNullOrWhiteSpace([string]$VoiceDocument.whisper_model_directory) -or
    -not (Test-CyrusLocalDriveAbsolutePath ([string]$VoiceDocument.whisper_model_directory)) -or
    -not (Test-Path -LiteralPath ([string]$VoiceDocument.whisper_model_directory) -PathType Container) -or
    [string]::IsNullOrWhiteSpace([string]$VoiceDocument.vad.artifact.path) -or
    -not (Test-CyrusLocalDriveAbsolutePath ([string]$VoiceDocument.vad.artifact.path)) -or
    -not (Test-Path -LiteralPath ([string]$VoiceDocument.vad.artifact.path) -PathType Leaf)
) {
    throw "windows_source.voice_artifacts_not_materialized"
}
Assert-CyrusNoReparsePath `
    -LiteralPath ([string]$VoiceDocument.whisper_model_directory) `
    -RequireContainer | Out-Null
Assert-CyrusNoReparsePath `
    -LiteralPath ([string]$VoiceDocument.vad.artifact.path) `
    -RequireLeaf | Out-Null
Assert-CyrusPathOutsideRepository `
    -LiteralPath ([string]$VoiceDocument.whisper_model_directory) `
    -Repository $Repository
Assert-CyrusPathOutsideRepository `
    -LiteralPath ([string]$VoiceDocument.vad.artifact.path) `
    -Repository $Repository
Invoke-CyrusCheckedCommand $VoicePython @(
    "-I", "-m", "jarvis_voice", "doctor", "--config", $VoiceConfigPath
) "windows_source.voice_doctor_blocked"
Invoke-CyrusCheckedCommand $InferencePython @(
    "-I", "-m", "jarvis_inference", "verify-manifest", "--manifest", $InferenceManifestPath
) "windows_source.inference_manifest_invalid"
Invoke-CyrusCheckedCommand $InferencePython @(
    "-I", "-c", "import importlib.metadata as m; assert m.version('jarvis-inference') == '1.0.0' and m.version('openai') == '2.46.0'"
) "windows_source.qualified_inference_environment_missing"

foreach ($RequiredEnvironmentName in @(
    "JARVIS_GATEWAY_WHEEL",
    "JARVIS_QUAL_GPU_DEVICE",
    "JARVIS_QUAL_GPU_MEMORY_BYTES",
    "JARVIS_QUAL_GPU_DRIVER",
    "JARVIS_QUAL_COMPUTE_RUNTIME",
    "JARVIS_QUAL_COMPUTE_CAPABILITY"
)) {
    if ([string]::IsNullOrWhiteSpace([Environment]::GetEnvironmentVariable($RequiredEnvironmentName))) {
        throw "windows_source.inference_runtime_environment_incomplete"
    }
}

# Artifact verification stays available before qualification. This second,
# content-free check is intentionally later because runtime identity variables
# must already be complete; it fails immediately instead of starting an
# inference supervisor that can never become ready.
Invoke-CyrusCheckedCommand $InferencePython @(
    "-I", "-m", "jarvis_inference", "verify-readiness", "--manifest", $InferenceManifestPath
) "windows_source.inference_qualification_not_ready"

[PSCustomObject]@{
    schema = "cyrus.windows-source-preflight/1"
    ready = $true
    source_run_only = $true
    production_ready = $false
    native_qualification_complete = $false
    profile = $Profile
    configs_materialized = $true
    model_artifacts_verified = $true
    inference_bearer_validated = $true
    network_used = $false
    secret_created_or_modified = $false
} | ConvertTo-Json -Compress
}
finally {
    try {
        if ($null -ne $PrivateRuntimeDirectory) {
            $CheckedRuntimeDirectory = Assert-CyrusNoReparsePath `
                -LiteralPath $PrivateRuntimeDirectory `
                -RequireContainer
            Remove-Item `
                -LiteralPath $CheckedRuntimeDirectory `
                -Recurse `
                -Force `
                -ErrorAction Stop
        }
    }
    finally {
        Restore-CyrusProcessEnvironmentSnapshot -Snapshot $OriginalProcessEnvironment
    }
}
