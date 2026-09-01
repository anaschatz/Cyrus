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
$Sleeper = Join-Path ([Environment]::SystemDirectory) "PING.EXE"
if (-not (Test-Path -LiteralPath $Sleeper -PathType Leaf)) {
    throw "windows_source.job_smoke_sleeper_missing"
}

# The native sleeper has no shell boundary. As a normal descendant it inherits
# the parent's Job Object membership automatically.
$StartInfo = New-Object Diagnostics.ProcessStartInfo
$StartInfo.FileName = $Sleeper
$StartInfo.Arguments = "-n 300 127.0.0.1"
$StartInfo.UseShellExecute = $false
$Child = [Diagnostics.Process]::Start($StartInfo)
if ($null -eq $Child) {
    throw "windows_source.job_smoke_child_start_failed"
}
$ChildId = $Child.Id
$Child.Dispose()
$Bytes = [Text.Encoding]::ASCII.GetBytes([string]$ChildId)
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

# Remain alive until the supervisor models an abrupt parent failure through
# its retained authoritative process handle.
Start-Sleep -Seconds 300
