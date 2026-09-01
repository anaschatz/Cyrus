[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $HandoffManifest,
    [switch] $Offline
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
. (Join-Path $PSScriptRoot "windows_source_common.ps1")

$Repository = Get-CyrusRepositoryRoot
$Toolchain = Assert-CyrusNativeWindowsToolchain
Set-CyrusNoModelNetworkEnvironment
Assert-CyrusTrackedTreeClean -Repository $Repository -IncludeUntracked
$FrozenCommit = Get-CyrusFrozenCommit -Repository $Repository
Assert-CyrusHandoffManifest `
    -Repository $Repository `
    -PythonExecutable $Toolchain.PythonExecutable `
    -HandoffManifest $HandoffManifest `
    -ExpectedCommit $FrozenCommit

$UseOffline = $Offline -or $env:UV_OFFLINE -eq "1"
$RootLockArguments = @("lock", "--check", "--project", $Repository)
$InferenceDirectory = Join-Path $Repository "inference"
$InferenceLockArguments = @("lock", "--check", "--project", $InferenceDirectory)
$OrchestratorDirectory = Join-Path $Repository "orchestrator"
$OrchestratorLockArguments = @("lock", "--check", "--project", $OrchestratorDirectory)
if ($UseOffline) {
    $RootLockArguments += "--offline"
    $InferenceLockArguments += "--offline"
    $OrchestratorLockArguments += "--offline"
}
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $RootLockArguments "windows_source.voice_lock_invalid"
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $InferenceLockArguments "windows_source.inference_lock_invalid"
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $OrchestratorLockArguments "windows_source.orchestrator_lock_invalid"

$VoiceSyncArguments = @(
    "sync"
    "--project", $Repository
    "--locked"
    "--no-editable"
    "--python", $Toolchain.PythonExecutable
    "--extra", "asr"
    "--extra", "audio"
    "--extra", "vad"
    "--extra", "transport"
    "--extra", "windows"
    "--extra", "packaging"
)
$InferenceSyncArguments = @(
    "sync"
    "--project", $InferenceDirectory
    "--locked"
    "--no-editable"
    "--no-dev"
    "--python", $Toolchain.PythonExecutable
    "--extra", "qualification"
)
$OrchestratorSyncArguments = @(
    "sync"
    "--project", $OrchestratorDirectory
    "--locked"
    "--no-editable"
    "--group", "dev"
    "--python", $Toolchain.PythonExecutable
)
if ($UseOffline) {
    $VoiceSyncArguments += "--offline"
    $InferenceSyncArguments += "--offline"
    $OrchestratorSyncArguments += "--offline"
}
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $VoiceSyncArguments "windows_source.voice_sync_failed"
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $InferenceSyncArguments "windows_source.inference_sync_failed"
Invoke-CyrusCheckedCommand $Toolchain.UvExecutable $OrchestratorSyncArguments "windows_source.orchestrator_sync_failed"

Set-CyrusRuntimeOfflineEnvironment
$VoicePython = Resolve-CyrusRequiredFile `
    (Join-Path $Repository ".venv\Scripts\python.exe") `
    "windows_source.voice_environment_missing"
$InferencePython = Resolve-CyrusRequiredFile `
    (Join-Path $InferenceDirectory ".venv\Scripts\python.exe") `
    "windows_source.inference_environment_missing"
$OrchestratorPython = Resolve-CyrusRequiredFile `
    (Join-Path $OrchestratorDirectory ".venv\Scripts\python.exe") `
    "windows_source.orchestrator_environment_missing"
Invoke-CyrusCheckedCommand $VoicePython @(
    "-I", "-c", "import importlib.metadata as m,jarvis_voice,cryptography; assert m.version('pyinstaller') == '6.21.0'; print('voice_import_ready')"
) "windows_source.voice_import_failed"
Invoke-CyrusCheckedCommand $InferencePython @(
    "-I", "-c", "import importlib.metadata as m,jarvis_inference,openai; assert m.version('jarvis-inference') == '1.0.0' and m.version('openai') == '2.46.0'; print('inference_import_ready')"
) "windows_source.inference_import_failed"
Invoke-CyrusCheckedCommand $OrchestratorPython @(
    "-I", "-c", "import jarvis_orchestrator,anamnesis; print('orchestrator_import_ready')"
) "windows_source.orchestrator_import_failed"

[PSCustomObject]@{
    schema = "cyrus.windows-source-provision/1"
    ready = $true
    source_run_only = $true
    production_ready = $false
    native_qualification_complete = $false
    locked_sync_complete = $true
    isolated_inference_environment = $true
    inference_qualification_dependencies = $true
    packaging_dependencies = $true
    orchestrator_test_dependencies = $true
    model_download_performed = $false
    secret_created_or_modified = $false
    offline_sync = [bool]$UseOffline
} | ConvertTo-Json -Compress
