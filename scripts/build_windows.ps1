[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("QualificationRunners", "VoicePackage")]
    [string]$Stage,

    [string]$CudaRuntimeManifest,
    [string]$RunnerOutputDirectory,
    [string]$RunnerBuildManifest
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Assert-BootstrapNoReparsePath {
    param(
        [Parameter(Mandatory = $true)] [string] $LiteralPath,
        [switch] $RequireLeaf
    )
    if ($LiteralPath -notmatch "^[A-Za-z]:[\\/]") {
        throw "windows_build.bootstrap_path_invalid"
    }
    $FullPath = [IO.Path]::GetFullPath($LiteralPath)
    $Root = [IO.Path]::GetPathRoot($FullPath)
    $Drive = [IO.DriveInfo]::new($Root)
    if ($Drive.DriveType -eq [IO.DriveType]::Network) {
        throw "windows_build.bootstrap_path_network"
    }
    $Current = $Root
    foreach ($Component in $FullPath.Substring($Root.Length).Split(
        @([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar),
        [StringSplitOptions]::RemoveEmptyEntries
    )) {
        $Current = Join-Path $Current $Component
        $Item = Get-Item -LiteralPath $Current -Force -ErrorAction Stop
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "windows_build.bootstrap_path_reparse"
        }
    }
    if ($RequireLeaf -and -not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "windows_build.bootstrap_path_not_file"
    }
    return $FullPath
}

$CommonScript = Assert-BootstrapNoReparsePath `
    -LiteralPath ([IO.Path]::GetFullPath((Join-Path $PSScriptRoot "windows_source_common.ps1"))) `
    -RequireLeaf
. $CommonScript

$Repository = Get-CyrusRepositoryRoot
Set-Location $Repository

if ($env:OS -ne "Windows_NT") {
    throw "This build must run on native Windows."
}

foreach ($Name in @(
    "GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES", "GIT_CONFIG_SYSTEM", "GIT_CONFIG_GLOBAL",
    "GIT_CONFIG_NOSYSTEM", "GIT_CONFIG_COUNT"
)) {
    [Environment]::SetEnvironmentVariable($Name, $null, "Process")
}
$GitCommand = Get-Command git -CommandType Application -ErrorAction Stop
$GitExecutable = Assert-CyrusNoReparsePath `
    -LiteralPath $GitCommand.Source `
    -RequireLeaf
Assert-CyrusTrackedTreeClean -Repository $Repository -IncludeUntracked -GitExecutable $GitExecutable
$VoicePython = Assert-CyrusNoReparsePath `
    -LiteralPath ([IO.Path]::GetFullPath((Join-Path $Repository ".venv\Scripts\python.exe"))) `
    -RequireLeaf
$InferencePython = Assert-CyrusNoReparsePath `
    -LiteralPath ([IO.Path]::GetFullPath((Join-Path $Repository "inference\.venv\Scripts\python.exe"))) `
    -RequireLeaf
$OrchestratorPython = Assert-CyrusNoReparsePath `
    -LiteralPath ([IO.Path]::GetFullPath((Join-Path $Repository "orchestrator\.venv\Scripts\python.exe"))) `
    -RequireLeaf
$BuildGuard = Assert-CyrusNoReparsePath `
    -LiteralPath (Join-Path $Repository "scripts\windows_build_guard.py") `
    -RequireLeaf

# Isolated mode is mandatory, but the build process also strips ambient Python
# injection, provider credentials, proxies and toolkit search hints so child
# hooks cannot accidentally observe them.
foreach ($Name in @(
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE",
    "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PIP_TRUSTED_HOST",
    "UV_INDEX_URL", "UV_EXTRA_INDEX_URL", "UV_DEFAULT_INDEX",
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY", "AZURE_OPENAI_API_KEY", "AZURE_CLIENT_ID",
    "AZURE_TENANT_ID", "AZURE_CLIENT_SECRET", "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "AWS_PROFILE",
    "AWS_DEFAULT_PROFILE", "GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
    "CLOUDSDK_CONFIG", "GITHUB_TOKEN", "GH_TOKEN", "LLAMA_API_KEY",
    "JARVIS_INFERENCE_API_KEY", "JARVIS_EGRESS_PROXY_CREDENTIALS_FILE",
    "JARVIS_EGRESS_PROXY_CREDENTIALS_SOURCE", "JARVIS_RESEARCH_BEARER_TOKEN_FILE",
    "JARVIS_RESEARCH_BEARER_TOKEN_SOURCE", "JARVIS_RESEARCH_EGRESS_PROXY_URL",
    "CYRUS_ASR_QUALIFICATION_API_KEY", "CYRUS_PROFILE_QUAL_API_KEY",
    "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "CUDA_PATH", "CUDA_HOME",
    "CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST"
)) {
    [Environment]::SetEnvironmentVariable($Name, $null, "Process")
}
Set-CyrusRuntimeOfflineEnvironment
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONUTF8 = "1"

foreach ($PythonExecutable in @($VoicePython, $InferencePython, $OrchestratorPython)) {
    $Identity = & $PythonExecutable -I -c "import platform,struct,sys; print(f'{platform.python_implementation()}:{sys.version_info.major}.{sys.version_info.minor}:{platform.machine()}:{struct.calcsize(chr(80))*8}')"
    if ($LASTEXITCODE -ne 0 -or $Identity -notmatch "^CPython:3\.11:(AMD64|x86_64):64$") {
        throw "The release build requires provisioned CPython 3.11 x64 environments."
    }
}

$PyInstallerVersion = & $VoicePython -I -c "import PyInstaller; print(PyInstaller.__version__)"
if ($PyInstallerVersion -ne "6.21.0") {
    throw "Expected the pre-approved PyInstaller 6.21.0 environment. Nothing was installed."
}

$CollectorRuntime = & $VoicePython -I -c "import importlib.metadata as m; print(m.version('faster-whisper') + ':' + m.version('ctranslate2') + ':' + m.version('pyinstaller'))"
if ($LASTEXITCODE -ne 0 -or $CollectorRuntime -ne "1.2.1:4.8.1:6.21.0") {
    throw "Expected the locked Windows ASR collector runtime. Nothing was installed."
}

$InferenceVersion = & $InferencePython -I -c "import importlib.metadata as m; print(m.version('jarvis-inference') + ':' + m.version('openai'))"
if ($LASTEXITCODE -ne 0 -or $InferenceVersion -ne "1.0.0:2.46.0") {
    throw "Expected the locked Inference qualification environment. Nothing was installed."
}

# Resolve every source/build input through the stdlib-only fixed-drive and
# reparse-point guard before PyInstaller is allowed to read it.
$BuildInputFiles = @(
    $BuildGuard,
    (Join-Path $Repository "inference\packaging\windows_native_fixture_runner.spec"),
    (Join-Path $Repository "inference\packaging\runtime_hook_harden_dll.py"),
    (Join-Path $Repository "inference\src\jarvis_inference\private_temp.py"),
    (Join-Path $Repository "inference\src\jarvis_inference\windows_process.py"),
    (Join-Path $Repository "inference\scripts\windows_native_fixture_runner.py"),
    (Join-Path $Repository "inference\scripts\windows_primary_profile_runner.py"),
    (Join-Path $Repository "packaging\windows_asr_collector.spec"),
    (Join-Path $Repository "packaging\windows_asr_collector_entry.py"),
    (Join-Path $Repository "packaging\windows_primary_profile_runner.spec"),
    (Join-Path $Repository "packaging\runtime_hook_harden_dll.py"),
    (Join-Path $Repository "packaging\windows-manifest.xml"),
    (Join-Path $Repository "packaging\jarvis_voice.spec"),
    (Join-Path $Repository "packaging\windows_entry.py"),
    (Join-Path $Repository "src\jarvis_voice\windows\cuda_bundle.py")
)
$InputArguments = @("-I", $BuildGuard, "verify-inputs", "--directory", $Repository)
foreach ($InputFile in @($VoicePython, $InferencePython, $OrchestratorPython) + $BuildInputFiles) {
    $InputArguments += @("--file", $InputFile)
}
& $VoicePython @InputArguments
if ($LASTEXITCODE -ne 0) { throw "A Windows build input was rejected." }

& $VoicePython -m unittest discover -s tests -v
if ($LASTEXITCODE -ne 0) { throw "Voice tests failed." }
& $InferencePython -m unittest discover -s inference/tests -v
if ($LASTEXITCODE -ne 0) { throw "Inference tests failed." }
& $InferencePython .\inference\tests\sdk_conformance.py -v
if ($LASTEXITCODE -ne 0) { throw "Inference SDK conformance failed." }
& $OrchestratorPython -m unittest discover -s orchestrator/tests -v
if ($LASTEXITCODE -ne 0) { throw "Orchestrator tests failed." }
& $VoicePython -m compileall -q src tests packaging scripts orchestrator/src orchestrator/tests inference/src inference/tests
if ($LASTEXITCODE -ne 0) { throw "Source compilation failed." }

if ($Stage -eq "QualificationRunners") {
    if ([string]::IsNullOrWhiteSpace($CudaRuntimeManifest)) {
        throw "-CudaRuntimeManifest is required for QualificationRunners."
    }
    if (-not [IO.Path]::IsPathRooted($CudaRuntimeManifest)) {
        $CudaRuntimeManifest = Join-Path $Repository $CudaRuntimeManifest
    }
    $CudaRuntimeManifest = [IO.Path]::GetFullPath($CudaRuntimeManifest)
    Assert-CyrusPathOutsideRepository `
        -LiteralPath $CudaRuntimeManifest `
        -Repository $Repository
    & $VoicePython -I $BuildGuard verify-cuda --manifest $CudaRuntimeManifest
    if ($LASTEXITCODE -ne 0) { throw "The private CUDA runtime bundle was rejected." }

    if ([string]::IsNullOrWhiteSpace($RunnerOutputDirectory)) {
        throw "-RunnerOutputDirectory outside the repository is required."
    }
    if (-not [IO.Path]::IsPathRooted($RunnerOutputDirectory)) {
        $RunnerOutputDirectory = Join-Path $Repository $RunnerOutputDirectory
    }
    if (Test-Path -LiteralPath $RunnerOutputDirectory) {
        throw "RunnerOutputDirectory must be a new directory; sealed runner bytes are never overwritten."
    }
    $RunnerParent = Split-Path -Parent $RunnerOutputDirectory
    Assert-CyrusPathOutsideRepository `
        -LiteralPath $RunnerParent `
        -Repository $Repository
    & $VoicePython -I $BuildGuard verify-inputs --directory $RunnerParent
    if ($LASTEXITCODE -ne 0) { throw "The runner output parent was rejected." }
    $RunnerOutputDirectory = (New-Item -ItemType Directory -Path $RunnerOutputDirectory).FullName
    & $VoicePython -I $BuildGuard verify-inputs --directory $RunnerOutputDirectory
    if ($LASTEXITCODE -ne 0) { throw "The runner output directory was rejected." }

    if ([string]::IsNullOrWhiteSpace($RunnerBuildManifest)) {
        $RunnerBuildManifest = Join-Path $RunnerOutputDirectory "qualification-runners-manifest.json"
    } elseif (-not [IO.Path]::IsPathRooted($RunnerBuildManifest)) {
        $RunnerBuildManifest = Join-Path $Repository $RunnerBuildManifest
    }
    if ((Split-Path -Parent $RunnerBuildManifest) -ne $RunnerOutputDirectory) {
        throw "RunnerBuildManifest must be created beside the sealed runners."
    }
    if (Test-Path -LiteralPath $RunnerBuildManifest) {
        throw "RunnerBuildManifest already exists."
    }

    $RunnerWorkDirectory = Join-Path $RunnerOutputDirectory ".pyinstaller-work"

    & $VoicePython -I -m PyInstaller --clean --noconfirm `
        --distpath $RunnerOutputDirectory `
        --workpath (Join-Path $RunnerWorkDirectory "native-fixture") `
        .\inference\packaging\windows_native_fixture_runner.spec
    if ($LASTEXITCODE -ne 0) { throw "Windows native Inference fixture runner build failed." }

    $PreviousCudaManifest = [Environment]::GetEnvironmentVariable(
        "CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST", "Process"
    )
    try {
        [Environment]::SetEnvironmentVariable(
            "CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST", $CudaRuntimeManifest, "Process"
        )
        & $VoicePython -I -m PyInstaller --clean --noconfirm `
            --distpath $RunnerOutputDirectory `
            --workpath (Join-Path $RunnerWorkDirectory "asr-collector") `
            .\packaging\windows_asr_collector.spec
        if ($LASTEXITCODE -ne 0) { throw "Windows ASR qualification collector build failed." }
    } finally {
        [Environment]::SetEnvironmentVariable(
            "CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST", $PreviousCudaManifest, "Process"
        )
    }

    & $VoicePython -I -m PyInstaller --clean --noconfirm `
        --distpath $RunnerOutputDirectory `
        --workpath (Join-Path $RunnerWorkDirectory "primary-profile") `
        .\packaging\windows_primary_profile_runner.spec
    if ($LASTEXITCODE -ne 0) { throw "Windows primary-profile fixture runner build failed." }

    $NativeFixtureRunner = Join-Path $RunnerOutputDirectory "native-fixture-runner.exe"
    $AsrCollector = Join-Path $RunnerOutputDirectory "cyrus-windows-asr-collector.exe"
    $PrimaryProfileRunner = Join-Path $RunnerOutputDirectory "cyrus-windows-primary-profile-runner.exe"
    foreach ($Runner in @($NativeFixtureRunner, $AsrCollector, $PrimaryProfileRunner)) {
        if (-not (Test-Path -LiteralPath $Runner -PathType Leaf)) {
            throw "A required Windows qualification runner was not produced."
        }
    }

    $SmokeDirectory = Join-Path $RunnerOutputDirectory ("boot-smoke-" + [Guid]::NewGuid().ToString("N"))
    $SmokeDirectory = (New-Item -ItemType Directory -Path $SmokeDirectory).FullName
    try {
        & $VoicePython -I $BuildGuard verify-inputs --directory $SmokeDirectory
        if ($LASTEXITCODE -ne 0) { throw "The onefile smoke directory was rejected." }
        $SmokeArguments = @(
            "-I", $BuildGuard, "boot-smoke",
            "--temporary-directory", $SmokeDirectory
        )
        foreach ($Runner in @($NativeFixtureRunner, $AsrCollector, $PrimaryProfileRunner)) {
            $SmokeArguments += @("--runner", $Runner)
        }
        & $VoicePython @SmokeArguments
        if ($LASTEXITCODE -ne 0) {
            throw "A Job-contained onefile runner boot smoke failed."
        }
    } finally {
        Remove-Item -LiteralPath $SmokeDirectory -Recurse -Force -ErrorAction SilentlyContinue
    }
    Remove-Item -LiteralPath $RunnerWorkDirectory -Recurse -Force -ErrorAction SilentlyContinue

    & $VoicePython -I $BuildGuard seal-runners `
        --directory $RunnerOutputDirectory `
        --cuda-manifest $CudaRuntimeManifest `
        --output $RunnerBuildManifest
    if ($LASTEXITCODE -ne 0) { throw "The qualification runner seal failed." }
    & $VoicePython -I $BuildGuard verify-runners --manifest $RunnerBuildManifest
    if ($LASTEXITCODE -ne 0) { throw "The qualification runner seal did not verify." }

    Write-Host "Sealed qualification runners created at: $RunnerOutputDirectory"
    Write-Host "Runner identity manifest created at: $RunnerBuildManifest"
    Write-Host "The private CUDA/cuDNN bundle was embedded only after exact offline validation; no runtime was downloaded."
    exit 0
}

if ([string]::IsNullOrWhiteSpace($RunnerBuildManifest)) {
    throw "-RunnerBuildManifest is required for VoicePackage."
}
if (-not [IO.Path]::IsPathRooted($RunnerBuildManifest)) {
    $RunnerBuildManifest = Join-Path $Repository $RunnerBuildManifest
}
$RunnerBuildManifest = [IO.Path]::GetFullPath($RunnerBuildManifest)
Assert-CyrusPathOutsideRepository `
    -LiteralPath $RunnerBuildManifest `
    -Repository $Repository
& $VoicePython -I $BuildGuard verify-runners --manifest $RunnerBuildManifest
if ($LASTEXITCODE -ne 0) {
    throw "The previously sealed qualification runners no longer match their manifest."
}

# The production package is a later stage.  It verifies, but never rebuilds or
# mutates, the already-qualified runner bytes.
$VoiceBuildDirectory = Join-Path $Repository "build"
$VoiceDistributionRoot = Join-Path $Repository "dist"
foreach ($ExistingOutput in @($VoiceBuildDirectory, $VoiceDistributionRoot)) {
    if (Test-Path -LiteralPath $ExistingOutput) {
        & $VoicePython -I $BuildGuard verify-inputs --directory $ExistingOutput
        if ($LASTEXITCODE -ne 0) {
            throw "An existing Voice package output directory was rejected."
        }
    }
}
& $VoicePython -I -m PyInstaller --clean --noconfirm .\packaging\jarvis_voice.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller build failed." }

$Distribution = Join-Path $VoiceDistributionRoot "JARVIS Voice"
& $VoicePython -I -m jarvis_voice dist-manifest --directory $Distribution
if ($LASTEXITCODE -ne 0) { throw "Distribution manifest creation failed." }
& $VoicePython -I -m jarvis_voice verify-dist --directory $Distribution
if ($LASTEXITCODE -ne 0) { throw "Distribution verification failed." }
& $VoicePython -I $BuildGuard verify-runners --manifest $RunnerBuildManifest
if ($LASTEXITCODE -ne 0) {
    throw "The sealed qualification runners changed during Voice packaging."
}

Write-Host "Unsigned development onedir created at: $Distribution"
Write-Host "Verified sealed qualification runner manifest: $RunnerBuildManifest"
Write-Host "No runner was rebuilt and no installer, model, certificate or signing secret was used."
