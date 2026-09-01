$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$script:CyrusRequiredUvVersion = "0.9.18"

# A source-run supervisor must own descendants even if an immediate parent
# exits before PowerShell reaches cleanup.  Create every top-level process
# suspended, place it in one KILL_ON_JOB_CLOSE Job Object, then resume it.  A
# post-start AssignProcessToJobObject call would leave a spawn/exit race.
function Initialize-CyrusWindowsSourceNativeTypes {
if ($null -eq ("Cyrus.WindowsSource.ProcessJob" -as [type])) {
    Add-Type -Language CSharp -TypeDefinition @'
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;

namespace Cyrus.WindowsSource
{
    // This wrapper never re-opens a process by PID.  The kernel handle returned by CreateProcessW
    // remains authoritative until Dispose, so status, waiting and fallback
    // termination cannot accidentally target a reused PID.
    public sealed class OwnedProcess : IDisposable
    {
        private const UInt32 WAIT_OBJECT_0 = 0x00000000;
        private const UInt32 WAIT_TIMEOUT = 0x00000102;
        private const UInt32 STILL_ACTIVE = 259;
        private const UInt32 WAIT_MILLISECONDS = 5000;
        private readonly object sync = new object();
        private IntPtr processHandle;

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern UInt32 WaitForSingleObject(IntPtr handle, UInt32 milliseconds);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool GetExitCodeProcess(IntPtr process, out UInt32 exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool TerminateProcess(IntPtr process, UInt32 exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        internal OwnedProcess(IntPtr ownedHandle, UInt32 processId)
        {
            if (ownedHandle == IntPtr.Zero || processId == 0)
                throw new InvalidOperationException("windows_source.process_handle_invalid");
            processHandle = ownedHandle;
            Id = processId;
        }

        // Id is diagnostic metadata only.  No operation in this class uses it
        // to rediscover or terminate a process.
        public UInt32 Id { get; private set; }

        public bool HasExited
        {
            get
            {
                lock (sync)
                {
                    EnsureOpen();
                    UInt32 result = WaitForSingleObject(processHandle, 0);
                    if (result == WAIT_OBJECT_0)
                        return true;
                    if (result == WAIT_TIMEOUT)
                        return false;
                    throw NativeFailure("windows_source.process_wait_failed");
                }
            }
        }

        public UInt32 ExitCode
        {
            get
            {
                lock (sync)
                {
                    EnsureOpen();
                    UInt32 exitCode;
                    if (!GetExitCodeProcess(processHandle, out exitCode))
                        throw NativeFailure("windows_source.process_exit_code_failed");
                    if (exitCode == STILL_ACTIVE &&
                        WaitForSingleObject(processHandle, 0) == WAIT_TIMEOUT)
                        throw new InvalidOperationException("windows_source.process_still_active");
                    return exitCode;
                }
            }
        }

        public bool WaitForExit(Int32 milliseconds)
        {
            if (milliseconds < 0)
                throw new ArgumentOutOfRangeException("milliseconds");
            lock (sync)
            {
                EnsureOpen();
                UInt32 result = WaitForSingleObject(processHandle, checked((UInt32)milliseconds));
                if (result == WAIT_OBJECT_0)
                    return true;
                if (result == WAIT_TIMEOUT)
                    return false;
                throw NativeFailure("windows_source.process_wait_failed");
            }
        }

        public void Terminate()
        {
            lock (sync)
            {
                EnsureOpen();
                UInt32 state = WaitForSingleObject(processHandle, 0);
                if (state == WAIT_OBJECT_0)
                    return;
                if (state != WAIT_TIMEOUT)
                    throw NativeFailure("windows_source.process_wait_failed");
                if (!TerminateProcess(processHandle, 1))
                {
                    Int32 error = Marshal.GetLastWin32Error();
                    // The process may have exited between the zero-time wait
                    // and TerminateProcess.  Only the retained handle decides.
                    if (WaitForSingleObject(processHandle, 0) != WAIT_OBJECT_0)
                        throw new Win32Exception(error, "windows_source.process_terminate_failed");
                }
                UInt32 waited = WaitForSingleObject(processHandle, WAIT_MILLISECONDS);
                if (waited == WAIT_TIMEOUT)
                    throw new InvalidOperationException("windows_source.process_terminate_timeout");
                if (waited != WAIT_OBJECT_0)
                    throw NativeFailure("windows_source.process_wait_failed");
            }
        }

        public void Dispose()
        {
            lock (sync)
            {
                if (processHandle == IntPtr.Zero)
                    return;
                if (!CloseHandle(processHandle))
                    throw NativeFailure("windows_source.process_handle_close_failed");
                processHandle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~OwnedProcess()
        {
            lock (sync)
            {
                if (processHandle != IntPtr.Zero)
                {
                    CloseHandle(processHandle);
                    processHandle = IntPtr.Zero;
                }
            }
        }

        private void EnsureOpen()
        {
            if (processHandle == IntPtr.Zero)
                throw new ObjectDisposedException("OwnedProcess");
        }

        private static Win32Exception NativeFailure(string code)
        {
            return new Win32Exception(Marshal.GetLastWin32Error(), code);
        }
    }

    public sealed class ProcessJob : IDisposable
    {
        private const UInt32 CREATE_SUSPENDED = 0x00000004;
        private const UInt32 CREATE_UNICODE_ENVIRONMENT = 0x00000400;
        private const UInt32 JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
        private const Int32 JobObjectExtendedLimitInformation = 9;
        private const UInt32 WAIT_MILLISECONDS = 5000;
        private readonly object sync = new object();
        private IntPtr jobHandle;

        [StructLayout(LayoutKind.Sequential)]
        private struct STARTUPINFO
        {
            public UInt32 cb;
            public IntPtr lpReserved;
            public IntPtr lpDesktop;
            public IntPtr lpTitle;
            public UInt32 dwX;
            public UInt32 dwY;
            public UInt32 dwXSize;
            public UInt32 dwYSize;
            public UInt32 dwXCountChars;
            public UInt32 dwYCountChars;
            public UInt32 dwFillAttribute;
            public UInt32 dwFlags;
            public UInt16 wShowWindow;
            public UInt16 cbReserved2;
            public IntPtr lpReserved2;
            public IntPtr hStdInput;
            public IntPtr hStdOutput;
            public IntPtr hStdError;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct PROCESS_INFORMATION
        {
            public IntPtr hProcess;
            public IntPtr hThread;
            public UInt32 dwProcessId;
            public UInt32 dwThreadId;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_BASIC_LIMIT_INFORMATION
        {
            public Int64 PerProcessUserTimeLimit;
            public Int64 PerJobUserTimeLimit;
            public UInt32 LimitFlags;
            public UIntPtr MinimumWorkingSetSize;
            public UIntPtr MaximumWorkingSetSize;
            public UInt32 ActiveProcessLimit;
            public UIntPtr Affinity;
            public UInt32 PriorityClass;
            public UInt32 SchedulingClass;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct IO_COUNTERS
        {
            public UInt64 ReadOperationCount;
            public UInt64 WriteOperationCount;
            public UInt64 OtherOperationCount;
            public UInt64 ReadTransferCount;
            public UInt64 WriteTransferCount;
            public UInt64 OtherTransferCount;
        }

        [StructLayout(LayoutKind.Sequential)]
        private struct JOBOBJECT_EXTENDED_LIMIT_INFORMATION
        {
            public JOBOBJECT_BASIC_LIMIT_INFORMATION BasicLimitInformation;
            public IO_COUNTERS IoInfo;
            public UIntPtr ProcessMemoryLimit;
            public UIntPtr JobMemoryLimit;
            public UIntPtr PeakProcessMemoryUsed;
            public UIntPtr PeakJobMemoryUsed;
        }

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
        private static extern IntPtr CreateJobObjectW(IntPtr attributes, string name);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool SetInformationJobObject(
            IntPtr job,
            Int32 informationClass,
            ref JOBOBJECT_EXTENDED_LIMIT_INFORMATION information,
            UInt32 informationLength);

        [DllImport("kernel32.dll", CharSet = CharSet.Unicode, ExactSpelling = true, SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CreateProcessW(
            string applicationName,
            StringBuilder commandLine,
            IntPtr processAttributes,
            IntPtr threadAttributes,
            [MarshalAs(UnmanagedType.Bool)] bool inheritHandles,
            UInt32 creationFlags,
            IntPtr environment,
            string currentDirectory,
            ref STARTUPINFO startupInfo,
            out PROCESS_INFORMATION processInformation);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern UInt32 ResumeThread(IntPtr thread);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool TerminateProcess(IntPtr process, UInt32 exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool TerminateJobObject(IntPtr job, UInt32 exitCode);

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern UInt32 WaitForSingleObject(IntPtr handle, UInt32 milliseconds);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool CloseHandle(IntPtr handle);

        public ProcessJob()
        {
            IntPtr created = CreateJobObjectW(IntPtr.Zero, null);
            if (created == IntPtr.Zero)
                throw NativeFailure("windows_source.job_create_failed");
            try
            {
                JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits =
                    new JOBOBJECT_EXTENDED_LIMIT_INFORMATION();
                limits.BasicLimitInformation.LimitFlags =
                    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
                if (!SetInformationJobObject(
                    created,
                    JobObjectExtendedLimitInformation,
                    ref limits,
                    (UInt32)Marshal.SizeOf(typeof(JOBOBJECT_EXTENDED_LIMIT_INFORMATION))))
                    throw NativeFailure("windows_source.job_policy_failed");
                jobHandle = created;
                created = IntPtr.Zero;
            }
            finally
            {
                if (created != IntPtr.Zero)
                    CloseHandle(created);
            }
        }

        public OwnedProcess Start(
            string executable,
            string[] arguments,
            string workingDirectory,
            string[] environmentEntries)
        {
            if (String.IsNullOrWhiteSpace(executable) || !Path.IsPathRooted(executable) ||
                !File.Exists(executable) || String.IsNullOrWhiteSpace(workingDirectory) ||
                !Path.IsPathRooted(workingDirectory) || !Directory.Exists(workingDirectory))
                throw new InvalidOperationException("windows_source.process_path_invalid");
            if (arguments == null)
                throw new InvalidOperationException("windows_source.process_arguments_invalid");

            string environmentBlock = BuildEnvironmentBlock(environmentEntries);

            StringBuilder commandLine = new StringBuilder(QuoteArgument(executable));
            foreach (string argument in arguments)
            {
                if (argument == null || argument.IndexOf('\0') >= 0)
                    throw new InvalidOperationException("windows_source.process_argument_invalid");
                commandLine.Append(' ').Append(QuoteArgument(argument));
            }
            if (commandLine.Length >= 32767)
                throw new InvalidOperationException("windows_source.process_command_too_long");

            lock (sync)
            {
                if (jobHandle == IntPtr.Zero)
                    throw new ObjectDisposedException("ProcessJob");
                STARTUPINFO startup = new STARTUPINFO();
                startup.cb = (UInt32)Marshal.SizeOf(typeof(STARTUPINFO));
                PROCESS_INFORMATION created = new PROCESS_INFORMATION();
                bool processCreated = false;
                IntPtr environmentPointer = IntPtr.Zero;
                try
                {
                    environmentPointer = Marshal.StringToHGlobalUni(environmentBlock);
                    if (!CreateProcessW(
                        executable,
                        commandLine,
                        IntPtr.Zero,
                        IntPtr.Zero,
                        false,
                        CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT,
                        environmentPointer,
                        workingDirectory,
                        ref startup,
                        out created))
                        throw NativeFailure("windows_source.process_create_failed");
                    processCreated = true;
                    if (!AssignProcessToJobObject(jobHandle, created.hProcess))
                        throw NativeFailure("windows_source.process_job_assign_failed");
                    if (ResumeThread(created.hThread) == UInt32.MaxValue)
                        throw NativeFailure("windows_source.process_resume_failed");
                    OwnedProcess process = new OwnedProcess(created.hProcess, created.dwProcessId);
                    created.hProcess = IntPtr.Zero;
                    return process;
                }
                catch
                {
                    if (processCreated)
                    {
                        TerminateProcess(created.hProcess, 1);
                        WaitForSingleObject(created.hProcess, WAIT_MILLISECONDS);
                    }
                    throw;
                }
                finally
                {
                    if (created.hThread != IntPtr.Zero)
                        CloseHandle(created.hThread);
                    if (created.hProcess != IntPtr.Zero)
                        CloseHandle(created.hProcess);
                    if (environmentPointer != IntPtr.Zero)
                    {
                        for (Int32 index = 0; index <= environmentBlock.Length; index++)
                            Marshal.WriteInt16(environmentPointer, index * 2, 0);
                        Marshal.FreeHGlobal(environmentPointer);
                    }
                }
            }
        }

        public void TerminateAll()
        {
            lock (sync)
            {
                if (jobHandle == IntPtr.Zero)
                    throw new ObjectDisposedException("ProcessJob");
                if (!TerminateJobObject(jobHandle, 1))
                    throw NativeFailure("windows_source.job_terminate_failed");
            }
        }

        public void Dispose()
        {
            lock (sync)
            {
                if (jobHandle == IntPtr.Zero)
                    return;
                if (!CloseHandle(jobHandle))
                    throw NativeFailure("windows_source.job_close_failed");
                jobHandle = IntPtr.Zero;
            }
            GC.SuppressFinalize(this);
        }

        ~ProcessJob()
        {
            lock (sync)
            {
                if (jobHandle != IntPtr.Zero)
                {
                    CloseHandle(jobHandle);
                    jobHandle = IntPtr.Zero;
                }
            }
        }

        private static Win32Exception NativeFailure(string code)
        {
            return new Win32Exception(Marshal.GetLastWin32Error(), code);
        }

        private static string BuildEnvironmentBlock(string[] environmentEntries)
        {
            if (environmentEntries == null || environmentEntries.Length == 0)
                throw new InvalidOperationException("windows_source.process_environment_invalid");
            SortedDictionary<string, string> values =
                new SortedDictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string entry in environmentEntries)
            {
                if (String.IsNullOrEmpty(entry) || entry.IndexOf('\0') >= 0)
                    throw new InvalidOperationException("windows_source.process_environment_invalid");
                Int32 separator = entry.IndexOf('=');
                if (separator <= 0)
                    throw new InvalidOperationException("windows_source.process_environment_invalid");
                string name = entry.Substring(0, separator);
                if (!IsEnvironmentName(name) || values.ContainsKey(name))
                    throw new InvalidOperationException("windows_source.process_environment_invalid");
                values.Add(name, entry.Substring(separator + 1));
            }
            StringBuilder block = new StringBuilder();
            foreach (KeyValuePair<string, string> item in values)
                block.Append(item.Key).Append('=').Append(item.Value).Append('\0');
            // StringToHGlobalUni contributes the second terminal NUL.
            if (block.Length == 0 || block.Length >= 32767)
                throw new InvalidOperationException("windows_source.process_environment_too_large");
            return block.ToString();
        }

        private static bool IsEnvironmentName(string name)
        {
            if (String.IsNullOrEmpty(name))
                return false;
            foreach (char character in name)
            {
                // Windows permits names such as ProgramFiles(x86).  The only
                // structural separators forbidden in an explicit block are
                // '=' and NUL; PowerShell supplies the production allowlist.
                if (character == '=' || character == '\0')
                    return false;
            }
            return true;
        }

        private static string QuoteArgument(string value)
        {
            StringBuilder result = new StringBuilder();
            result.Append('"');
            Int32 slashes = 0;
            foreach (char character in value)
            {
                if (character == '\\')
                {
                    slashes++;
                    continue;
                }
                if (character == '"')
                {
                    result.Append('\\', slashes * 2 + 1).Append('"');
                    slashes = 0;
                    continue;
                }
                result.Append('\\', slashes).Append(character);
                slashes = 0;
            }
            result.Append('\\', slashes * 2).Append('"');
            return result.ToString();
        }
    }
}
'@
}
}

function New-CyrusSourceProcessJob {
    if ($env:OS -ne "Windows_NT") {
        throw "windows_source.native_windows_required"
    }
    Initialize-CyrusWindowsSourceNativeTypes
    return New-Object -TypeName Cyrus.WindowsSource.ProcessJob
}

function Start-CyrusSourceJobProcess {
    param(
        [Parameter(Mandatory = $true)] [object] $Job,
        [Parameter(Mandatory = $true)] [string] $Executable,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [Parameter(Mandatory = $true)] [string] $WorkingDirectory,
        [Parameter(Mandatory = $true)] [string[]] $EnvironmentEntries
    )

    if (
        $null -eq $Job -or
        $Job.GetType().FullName -ne "Cyrus.WindowsSource.ProcessJob"
    ) {
        throw "windows_source.job_invalid"
    }
    $CheckedExecutable = Assert-CyrusNoReparsePath `
        -LiteralPath $Executable `
        -RequireLeaf
    $CheckedWorkingDirectory = Assert-CyrusNoReparsePath `
        -LiteralPath $WorkingDirectory `
        -RequireContainer
    foreach ($Argument in $Arguments) {
        if ($null -eq $Argument -or $Argument.Contains('"') -or $Argument.Contains("`0")) {
            throw "windows_source.process_argument_invalid"
        }
    }
    if ($EnvironmentEntries.Count -eq 0) {
        throw "windows_source.process_environment_invalid"
    }
    return $Job.Start(
        $CheckedExecutable,
        $Arguments,
        $CheckedWorkingDirectory,
        $EnvironmentEntries
    )
}

function New-CyrusSourceChildEnvironment {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("Validator", "Inference", "Orchestrator", "Voice")]
        [string] $Role,
        [Parameter(Mandatory = $true)] [string] $PrivateRuntimeDirectory,
        [string[]] $AdditionalPathDirectories = @(),
        [AllowEmptyString()] [string] $InferenceBearer = ""
    )

    # No ambient PATH, temp path, proxy credential, cloud/API token or Python
    # injection variable is copied.  OS paths come from Win32-backed .NET APIs
    # and every filesystem component is checked before it enters the block.
    $SystemDirectory = Assert-CyrusNoReparsePath `
        -LiteralPath ([Environment]::SystemDirectory) `
        -RequireContainer
    $WindowsDirectory = Assert-CyrusNoReparsePath `
        -LiteralPath (([IO.Directory]::GetParent($SystemDirectory)).FullName) `
        -RequireContainer
    $RuntimeDirectory = Assert-CyrusNoReparsePath `
        -LiteralPath $PrivateRuntimeDirectory `
        -RequireContainer
    $CommandProcessor = Assert-CyrusNoReparsePath `
        -LiteralPath (Join-Path $SystemDirectory "cmd.exe") `
        -RequireLeaf
    $SystemDriveRoot = [IO.Path]::GetPathRoot($WindowsDirectory)
    if ($SystemDriveRoot -notmatch "^[A-Za-z]:[\\/]$") {
        throw "windows_source.required_environment_missing"
    }

    $KnownFolders = @{
        "USERPROFILE" = [System.Environment+SpecialFolder]::UserProfile
        "APPDATA" = [System.Environment+SpecialFolder]::ApplicationData
        "LOCALAPPDATA" = [System.Environment+SpecialFolder]::LocalApplicationData
        "ProgramData" = [System.Environment+SpecialFolder]::CommonApplicationData
        "ProgramFiles" = [System.Environment+SpecialFolder]::ProgramFiles
        "ProgramFiles(x86)" = [System.Environment+SpecialFolder]::ProgramFilesX86
        "CommonProgramFiles" = [System.Environment+SpecialFolder]::CommonProgramFiles
        "CommonProgramFiles(x86)" = [System.Environment+SpecialFolder]::CommonProgramFilesX86
    }
    $Values = @{}
    foreach ($Name in $KnownFolders.Keys) {
        $Candidate = [Environment]::GetFolderPath($KnownFolders[$Name])
        if (-not [string]::IsNullOrWhiteSpace($Candidate)) {
            $Values[$Name] = Assert-CyrusNoReparsePath `
                -LiteralPath $Candidate `
                -RequireContainer
        }
    }
    foreach ($Name in @("USERPROFILE", "APPDATA", "LOCALAPPDATA", "ProgramData")) {
        if (-not $Values.ContainsKey($Name)) {
            throw "windows_source.required_environment_missing"
        }
    }

    $PathDirectories = @($SystemDirectory, $WindowsDirectory)
    foreach ($Candidate in @(
        (Join-Path $SystemDirectory "Wbem"),
        (Join-Path $SystemDirectory "WindowsPowerShell\v1.0")
    )) {
        if (Test-Path -LiteralPath $Candidate -PathType Container) {
            $PathDirectories += Assert-CyrusNoReparsePath `
                -LiteralPath $Candidate `
                -RequireContainer
        }
    }
    foreach ($Candidate in $AdditionalPathDirectories) {
        $CheckedPathDirectory = Assert-CyrusNoReparsePath `
            -LiteralPath $Candidate `
            -RequireContainer
        if ($CheckedPathDirectory -notin $PathDirectories) {
            $PathDirectories += $CheckedPathDirectory
        }
    }
    $Values["SystemRoot"] = $WindowsDirectory
    $Values["WINDIR"] = $WindowsDirectory
    $Values["SystemDrive"] = $SystemDriveRoot.Substring(0, 2)
    $Values["ComSpec"] = $CommandProcessor
    $Values["PATH"] = [string]::Join(";", [string[]]$PathDirectories)
    $Values["PATHEXT"] = ".COM;.EXE;.BAT;.CMD"
    $Values["TEMP"] = $RuntimeDirectory
    $Values["TMP"] = $RuntimeDirectory
    $Values["OS"] = "Windows_NT"
    $Values["PROCESSOR_ARCHITECTURE"] = "AMD64"
    $Values["NUMBER_OF_PROCESSORS"] = ([Environment]::ProcessorCount).ToString(
        [Globalization.CultureInfo]::InvariantCulture
    )
    $UserRoot = [IO.Path]::GetPathRoot($Values["USERPROFILE"])
    if ($UserRoot -match "^[A-Za-z]:[\\/]$") {
        $Values["HOMEDRIVE"] = $UserRoot.Substring(0, 2)
        $Values["HOMEPATH"] = $Values["USERPROFILE"].Substring(2)
    }
    if ($Values.ContainsKey("ProgramFiles")) {
        $Values["ProgramW6432"] = $Values["ProgramFiles"]
    }
    if ($Values.ContainsKey("CommonProgramFiles")) {
        $Values["CommonProgramW6432"] = $Values["CommonProgramFiles"]
    }

    # Generated policy values are the only Python/network-related entries.
    $Values["UV_PYTHON_DOWNLOADS"] = "never"
    $Values["UV_NO_PROGRESS"] = "1"
    $Values["UV_OFFLINE"] = "1"
    $Values["HF_HUB_OFFLINE"] = "1"
    $Values["TRANSFORMERS_OFFLINE"] = "1"
    $Values["HF_DATASETS_OFFLINE"] = "1"
    $Values["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    $Values["PIP_NO_INDEX"] = "1"
    $Values["NO_PROXY"] = "127.0.0.1,localhost"
    $Values["PYTHONNOUSERSITE"] = "1"
    $Values["PYTHONDONTWRITEBYTECODE"] = "1"
    $Values["PYTHONUTF8"] = "1"

    if ($Role -eq "Inference") {
        if (
            [string]::IsNullOrWhiteSpace($InferenceBearer) -or
            $InferenceBearer.Length -gt 4096 -or
            [Text.Encoding]::UTF8.GetByteCount($InferenceBearer) -lt 32 -or
            $InferenceBearer.Contains("`0")
        ) {
            throw "windows_source.inference_auth_invalid"
        }
        $Values["JARVIS_INFERENCE_API_KEY"] = $InferenceBearer
        foreach ($Name in @(
            "JARVIS_GATEWAY_WHEEL",
            "JARVIS_QUAL_GPU_DEVICE",
            "JARVIS_QUAL_GPU_MEMORY_BYTES",
            "JARVIS_QUAL_GPU_DRIVER",
            "JARVIS_QUAL_COMPUTE_RUNTIME",
            "JARVIS_QUAL_COMPUTE_CAPABILITY"
        )) {
            $Value = [Environment]::GetEnvironmentVariable(
                $Name,
                [EnvironmentVariableTarget]::Process
            )
            if ([string]::IsNullOrWhiteSpace($Value) -or $Value.Contains("`0")) {
                throw "windows_source.inference_runtime_environment_incomplete"
            }
            $Values[$Name] = $Value
        }
    }
    elseif (-not [string]::IsNullOrEmpty($InferenceBearer)) {
        throw "windows_source.inference_auth_scope_invalid"
    }

    [string[]]$Entries = @(
        $Values.Keys |
            Sort-Object |
            ForEach-Object { "{0}={1}" -f $_, $Values[$_] }
    )
    return $Entries
}

function Get-CyrusProcessEnvironmentSnapshot {
    $Snapshot = @{}
    foreach ($Entry in [Environment]::GetEnvironmentVariables(
        [EnvironmentVariableTarget]::Process
    ).GetEnumerator()) {
        $Name = [string]$Entry.Key
        if (-not $Name.StartsWith("=", [StringComparison]::Ordinal)) {
            $Snapshot[$Name] = [string]$Entry.Value
        }
    }
    return $Snapshot
}

function Set-CyrusProcessEnvironmentEntries {
    param([Parameter(Mandatory = $true)] [string[]] $EnvironmentEntries)

    $Desired = @{}
    foreach ($Entry in $EnvironmentEntries) {
        if ([string]::IsNullOrEmpty($Entry) -or $Entry.Contains("`0")) {
            throw "windows_source.process_environment_invalid"
        }
        $Separator = $Entry.IndexOf("=")
        if ($Separator -le 0) {
            throw "windows_source.process_environment_invalid"
        }
        $Name = $Entry.Substring(0, $Separator)
        if ($Desired.ContainsKey($Name)) {
            throw "windows_source.process_environment_invalid"
        }
        $Desired[$Name] = $Entry.Substring($Separator + 1)
    }
    foreach ($Name in @(
        [Environment]::GetEnvironmentVariables(
            [EnvironmentVariableTarget]::Process
        ).Keys
    )) {
        if (([string]$Name).StartsWith("=", [StringComparison]::Ordinal)) {
            continue
        }
        [Environment]::SetEnvironmentVariable(
            [string]$Name,
            $null,
            [EnvironmentVariableTarget]::Process
        )
    }
    foreach ($Name in $Desired.Keys) {
        [Environment]::SetEnvironmentVariable(
            [string]$Name,
            [string]$Desired[$Name],
            [EnvironmentVariableTarget]::Process
        )
    }
}

function Restore-CyrusProcessEnvironmentSnapshot {
    param([Parameter(Mandatory = $true)] [hashtable] $Snapshot)

    [string[]]$Entries = @(
        $Snapshot.Keys |
            ForEach-Object { "{0}={1}" -f $_, $Snapshot[$_] }
    )
    Set-CyrusProcessEnvironmentEntries -EnvironmentEntries $Entries
}

function Get-CyrusRepositoryRoot {
    $Candidate = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
    return Assert-CyrusNoReparsePath -LiteralPath $Candidate -RequireContainer
}

function Test-CyrusLocalDriveAbsolutePath {
    param([AllowEmptyString()] [string] $LiteralPath)

    # Path.IsPathFullyQualified was added after .NET Framework and therefore
    # is unavailable in Windows PowerShell 5.1.  The source-run contract is
    # narrower anyway: only an absolute local drive-letter path is admissible;
    # Assert-CyrusNoReparsePath performs the remaining drive and component
    # checks before use.
    return (
        -not [string]::IsNullOrWhiteSpace($LiteralPath) -and
        $LiteralPath -match "^[A-Za-z]:[\\/]"
    )
}

function Assert-CyrusNativeWindowsToolchain {
    param(
        [AllowEmptyString()] [string] $PythonLauncherPath = "",
        [AllowEmptyString()] [string] $UvExecutablePath = ""
    )

    if ($env:OS -ne "Windows_NT") {
        throw "windows_source.native_windows_required"
    }

    if ([string]::IsNullOrWhiteSpace($PythonLauncherPath)) {
        $PythonLauncherCommand = Get-Command py `
            -CommandType Application `
            -ErrorAction SilentlyContinue
        if ($null -eq $PythonLauncherCommand) {
            throw "windows_source.python_launcher_missing"
        }
        $PythonLauncherPath = $PythonLauncherCommand.Source
    }
    $PythonLauncher = Assert-CyrusNoReparsePath `
        -LiteralPath $PythonLauncherPath `
        -RequireLeaf
    $PythonIdentityRaw = & $PythonLauncher -3.11 -I -c "import json,platform,struct,sys; print(json.dumps({'implementation':platform.python_implementation(),'major':sys.version_info.major,'minor':sys.version_info.minor,'bits':struct.calcsize('P')*8,'machine':platform.machine(),'executable':sys.executable},separators=(',',':')))"
    if ($LASTEXITCODE -ne 0) {
        throw "windows_source.python_311_missing"
    }
    try {
        $PythonIdentity = $PythonIdentityRaw | ConvertFrom-Json
    }
    catch {
        throw "windows_source.python_identity_invalid"
    }
    if (
        $PythonIdentity.implementation -ne "CPython" -or
        $PythonIdentity.major -ne 3 -or
        $PythonIdentity.minor -ne 11 -or
        $PythonIdentity.bits -ne 64 -or
        $PythonIdentity.machine -notin @("AMD64", "x86_64") -or
        [string]::IsNullOrWhiteSpace($PythonIdentity.executable)
    ) {
        throw "windows_source.python_311_x64_required"
    }

    if ([string]::IsNullOrWhiteSpace($UvExecutablePath)) {
        $UvCommand = Get-Command uv `
            -CommandType Application `
            -ErrorAction SilentlyContinue
        if ($null -eq $UvCommand) {
            throw "windows_source.uv_missing"
        }
        $UvExecutablePath = $UvCommand.Source
    }
    $UvExecutable = Assert-CyrusNoReparsePath `
        -LiteralPath $UvExecutablePath `
        -RequireLeaf
    $UvIdentity = (& $UvExecutable --version).Trim()
    if ($LASTEXITCODE -ne 0 -or $UvIdentity -notmatch "^uv 0\.9\.18(?:\s|$)") {
        throw "windows_source.uv_version_mismatch"
    }

    return [PSCustomObject]@{
        PythonExecutable = Assert-CyrusNoReparsePath `
            -LiteralPath ([IO.Path]::GetFullPath([string]$PythonIdentity.executable)) `
            -RequireLeaf
        UvExecutable = $UvExecutable
        UvVersion = $script:CyrusRequiredUvVersion
    }
}

function Set-CyrusNoModelNetworkEnvironment {
    $env:UV_PYTHON_DOWNLOADS = "never"
    $env:UV_NO_PROGRESS = "1"
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:HF_DATASETS_OFFLINE = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:NO_PROXY = "127.0.0.1,localhost"
}

function Set-CyrusRuntimeOfflineEnvironment {
    Set-CyrusNoModelNetworkEnvironment
    $env:UV_OFFLINE = "1"
    $env:PIP_NO_INDEX = "1"
}

function Get-CyrusFrozenCommit {
    param(
        [Parameter(Mandatory = $true)] [string] $Repository,
        [AllowEmptyString()] [string] $GitExecutable = ""
    )

    if ([string]::IsNullOrWhiteSpace($GitExecutable)) {
        $GitCommand = Get-Command git `
            -CommandType Application `
            -ErrorAction SilentlyContinue
        if ($null -eq $GitCommand) {
            throw "windows_source.git_missing"
        }
        $GitExecutable = $GitCommand.Source
    }
    $GitExecutable = Assert-CyrusNoReparsePath `
        -LiteralPath $GitExecutable `
        -RequireLeaf
    $Commit = (& $GitExecutable -C $Repository rev-parse --verify HEAD).Trim()
    if ($LASTEXITCODE -ne 0 -or $Commit -notmatch "^[0-9a-f]{40}$") {
        throw "windows_source.git_identity_invalid"
    }
    return $Commit
}

function Assert-CyrusTrackedTreeClean {
    param(
        [Parameter(Mandatory = $true)] [string] $Repository,
        [AllowEmptyString()] [string] $GitExecutable = "",
        [switch] $IncludeUntracked
    )

    $Arguments = @("-C", $Repository, "status", "--porcelain=v1")
    if ($IncludeUntracked) {
        $Arguments += "--untracked-files=all"
    }
    else {
        $Arguments += "--untracked-files=no"
    }
    if ([string]::IsNullOrWhiteSpace($GitExecutable)) {
        $GitCommand = Get-Command git `
            -CommandType Application `
            -ErrorAction SilentlyContinue
        if ($null -eq $GitCommand) {
            throw "windows_source.git_missing"
        }
        $GitExecutable = $GitCommand.Source
    }
    $GitExecutable = Assert-CyrusNoReparsePath `
        -LiteralPath $GitExecutable `
        -RequireLeaf
    $Status = & $GitExecutable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "windows_source.git_status_failed"
    }
    if ($null -ne $Status -and @($Status).Count -gt 0) {
        throw "windows_source.worktree_not_clean"
    }
}

function Resolve-CyrusRequiredFile {
    param(
        [Parameter(Mandatory = $true)] [string] $LiteralPath,
        [Parameter(Mandatory = $true)] [string] $FailureCode
    )

    if (-not (Test-CyrusLocalDriveAbsolutePath $LiteralPath)) {
        throw $FailureCode
    }
    try {
        $FullPath = [IO.Path]::GetFullPath($LiteralPath)
        $Resolved = Assert-CyrusNoReparsePath -LiteralPath $FullPath -RequireLeaf
    }
    catch {
        throw $FailureCode
    }
    # Resolution is safe only after every component has been inspected.  It
    # canonicalizes alternate path spellings (including legacy short names)
    # before containment comparisons without hiding a reparse point.
    return $Resolved
}

function Assert-CyrusNoReparsePath {
    param(
        [Parameter(Mandatory = $true)] [string] $LiteralPath,
        [switch] $RequireLeaf,
        [switch] $RequireContainer
    )

    if (
        -not (Test-CyrusLocalDriveAbsolutePath $LiteralPath) -or
        ($RequireLeaf -and $RequireContainer)
    ) {
        throw "windows_source.path_invalid"
    }
    $FullPath = [IO.Path]::GetFullPath($LiteralPath)
    $Root = [IO.Path]::GetPathRoot($FullPath)
    if (
        [string]::IsNullOrWhiteSpace($Root) -or
        $Root -notmatch "^[A-Za-z]:[\\/]$"
    ) {
        throw "windows_source.path_invalid"
    }
    try {
        $Drive = [IO.DriveInfo]::new($Root)
    }
    catch {
        throw "windows_source.path_invalid"
    }
    if ($Drive.DriveType -eq [IO.DriveType]::Network) {
        throw "windows_source.path_network_forbidden"
    }
    $Relative = $FullPath.Substring($Root.Length)
    $Components = $Relative.Split(
        @([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar),
        [StringSplitOptions]::RemoveEmptyEntries
    )
    $Current = $Root
    foreach ($Component in $Components) {
        $Current = Join-Path $Current $Component
        $Item = Get-Item -LiteralPath $Current -Force -ErrorAction Stop
        if (($Item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "windows_source.path_reparse_point"
        }
    }
    if ($RequireLeaf -and -not (Test-Path -LiteralPath $FullPath -PathType Leaf)) {
        throw "windows_source.path_not_file"
    }
    if ($RequireContainer -and -not (Test-Path -LiteralPath $FullPath -PathType Container)) {
        throw "windows_source.path_not_directory"
    }
    # Resolve only after the entire existing chain has passed the no-reparse
    # check, so alternate spellings cannot defeat later containment tests.
    return (Resolve-Path -LiteralPath $FullPath -ErrorAction Stop).ProviderPath
}

function Assert-CyrusPathOutsideRepository {
    param(
        [Parameter(Mandatory = $true)] [string] $LiteralPath,
        [Parameter(Mandatory = $true)] [string] $Repository
    )

    $FullPath = Assert-CyrusNoReparsePath -LiteralPath $LiteralPath
    $RepositoryPath = Assert-CyrusNoReparsePath -LiteralPath $Repository -RequireContainer
    $Separators = [char[]]@(
        [IO.Path]::DirectorySeparatorChar,
        [IO.Path]::AltDirectorySeparatorChar
    )
    $RepositoryPrefix = $RepositoryPath.TrimEnd($Separators) + [IO.Path]::DirectorySeparatorChar
    if (
        [string]::Equals(
            $FullPath.TrimEnd($Separators),
            $RepositoryPath.TrimEnd($Separators),
            [StringComparison]::OrdinalIgnoreCase
        ) -or
        $FullPath.StartsWith($RepositoryPrefix, [StringComparison]::OrdinalIgnoreCase)
    ) {
        throw "windows_source.private_path_inside_repository"
    }
}

function Invoke-CyrusCheckedCommand {
    param(
        [Parameter(Mandatory = $true)] [string] $Executable,
        [Parameter(Mandatory = $true)] [string[]] $Arguments,
        [Parameter(Mandatory = $true)] [string] $FailureCode
    )

    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw $FailureCode
    }
}

function Assert-CyrusHandoffManifest {
    param(
        [Parameter(Mandatory = $true)] [string] $Repository,
        [Parameter(Mandatory = $true)] [string] $PythonExecutable,
        [Parameter(Mandatory = $true)] [string] $HandoffManifest,
        [Parameter(Mandatory = $true)] [string] $ExpectedCommit
    )

    $Manifest = Resolve-CyrusRequiredFile $HandoffManifest "windows_source.handoff_manifest_missing"
    Invoke-CyrusCheckedCommand $PythonExecutable @(
        (Join-Path $Repository "scripts\verify_windows_handoff.py")
        "verify"
        "--manifest"
        $Manifest
        "--expected-commit"
        $ExpectedCommit
    ) "windows_source.handoff_manifest_invalid"
}

function Stop-CyrusOwnedProcess {
    param([object] $Process)

    if ($null -eq $Process) {
        return
    }
    if ($Process.GetType().FullName -ne "Cyrus.WindowsSource.OwnedProcess") {
        throw "windows_source.process_handle_invalid"
    }
    try {
        if (-not $Process.HasExited) {
            # This calls TerminateProcess through the retained CreateProcessW
            # handle.  PID-based rediscovery is deliberately unavailable.
            $Process.Terminate()
            [void]$Process.WaitForExit(5000)
        }
    }
    catch {
        # Cleanup is best effort; the caller still checks for owned survivors.
    }
}
