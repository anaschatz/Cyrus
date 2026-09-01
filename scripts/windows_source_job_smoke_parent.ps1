[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string] $ChildPidPath
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Windows PowerShell 5.1 runs on .NET Framework, which does not expose
# Path.IsPathFullyQualified.  This smoke accepts only the same absolute local
# drive-letter syntax as the supervising source-run contract.
if (
    [string]::IsNullOrWhiteSpace($ChildPidPath) -or
    $ChildPidPath -notmatch "^[A-Za-z]:[\\/]"
) {
    throw "windows_source.job_smoke_pid_path_invalid"
}
$PowerShell = Join-Path `
    ([Environment]::SystemDirectory) `
    "WindowsPowerShell\v1.0\powershell.exe"
if (-not (Test-Path -LiteralPath $PowerShell -PathType Leaf)) {
    throw "windows_source.job_smoke_shell_missing"
}

# The encoded command has no shell quoting boundary and lives long enough for
# the parent to exit first. As a normal descendant it inherits the parent's
# Job Object membership automatically.
$EncodedSleeper = [Convert]::ToBase64String(
    [Text.Encoding]::Unicode.GetBytes("Start-Sleep -Seconds 300")
)
$Child = Start-Process `
    -FilePath $PowerShell `
    -ArgumentList @(
        "-NoProfile",
        "-NonInteractive",
        "-EncodedCommand",
        $EncodedSleeper
    ) `
    -PassThru
$Bytes = [Text.Encoding]::ASCII.GetBytes([string]$Child.Id)
$Stream = [IO.File]::Open(
    $ChildPidPath,
    [IO.FileMode]::CreateNew,
    [IO.FileAccess]::Write,
    [IO.FileShare]::None
)
try {
    $Stream.Write($Bytes, 0, $Bytes.Length)
    $Stream.Flush($true)
}
finally {
    $Stream.Dispose()
}

# Force the smoke parent boundary after the durable PID handoff.
[Environment]::Exit(0)
