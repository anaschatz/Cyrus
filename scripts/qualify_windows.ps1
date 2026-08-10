param(
    [Parameter(Mandatory = $true)] [string] $Config,
    [Parameter(Mandatory = $true)] [string] $Samples,
    [Parameter(Mandatory = $true)] [string] $Output,
    [string] $Distribution
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
    throw "Release qualification must run on native Windows."
}

$Repository = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if (-not $Distribution) {
    $Distribution = Join-Path $Repository "dist\JARVIS Voice"
}
$DistributionPath = (Resolve-Path $Distribution).Path
$ConfigPath = (Resolve-Path $Config).Path
$SamplesPath = (Resolve-Path $Samples).Path
$OutputPath = [IO.Path]::GetFullPath($Output)
$OutputDirectory = [IO.Path]::GetDirectoryName($OutputPath)
if (-not (Test-Path -LiteralPath $OutputDirectory -PathType Container)) {
    throw "The qualification output directory must already exist."
}

$Executable = Join-Path $DistributionPath "JARVIS Voice.exe"
if (-not (Test-Path -LiteralPath $Executable -PathType Leaf)) {
    throw "The frozen JARVIS Voice executable is missing."
}

$FrozenDoctor = Join-Path $OutputDirectory "frozen-doctor.json"
& $Executable doctor --config $ConfigPath --output $FrozenDoctor
$FrozenDoctorExit = $LASTEXITCODE

$env:PYTHONPATH = (Resolve-Path (Join-Path $Repository "src")).Path
py -3.11 -m jarvis_voice qualify-release `
    --config $ConfigPath `
    --distribution $DistributionPath `
    --samples $SamplesPath `
    --frozen-doctor $FrozenDoctor `
    --output $OutputPath
$QualificationExit = $LASTEXITCODE

if ($FrozenDoctorExit -ne 0) {
    Write-Warning "The frozen doctor was blocked; the final report must remain ready=false."
}
if ($QualificationExit -eq 0) {
    Write-Host "Native Windows release qualification passed."
} else {
    Write-Warning "Release qualification is blocked. Inspect only the content-free codes in the report."
}
exit $QualificationExit
