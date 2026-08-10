$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repository

if ($env:OS -ne "Windows_NT") {
    throw "This build must run on native Windows."
}

$Architecture = py -3.11 -c "import platform,struct; print(platform.machine() + ':' + str(struct.calcsize('P') * 8))"
if ($Architecture -notmatch "^(AMD64|x86_64):64$") {
    throw "The release build requires CPython 3.11 x64."
}

$PyInstallerVersion = py -3.11 -c "import PyInstaller; print(PyInstaller.__version__)"
if ($PyInstallerVersion -ne "6.21.0") {
    throw "Expected the pre-approved PyInstaller 6.21.0 environment. Nothing was installed."
}

$env:PYTHONPATH = (Resolve-Path ".\src").Path
py -3.11 -m unittest discover -s tests -v
py -3.11 -m compileall -q src tests packaging
py -3.11 -m PyInstaller --clean --noconfirm .\packaging\jarvis_voice.spec

$Distribution = Join-Path $Repository "dist\JARVIS Voice"
py -3.11 -m jarvis_voice dist-manifest --directory $Distribution
py -3.11 -m jarvis_voice verify-dist --directory $Distribution

Write-Host "Unsigned development onedir created at: $Distribution"
Write-Host "No installer, model, VAD artifact, CUDA runtime, certificate or signing secret was used."
