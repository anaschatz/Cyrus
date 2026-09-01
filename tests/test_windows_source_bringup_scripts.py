from __future__ import annotations

import re
import tomllib
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "scripts" / "windows_source_common.ps1"
PROVISION = ROOT / "scripts" / "provision_windows_source.ps1"
PREFLIGHT = ROOT / "scripts" / "preflight_windows_source.ps1"
RUN = ROOT / "scripts" / "run_windows_source_beta.ps1"
JOB_SMOKE = ROOT / "scripts" / "test_windows_source_job.ps1"
BUILD = ROOT / "scripts" / "build_windows.ps1"
WORKFLOW = ROOT / ".github" / "workflows" / "backend-ci.yml"
INFERENCE_LOCK = ROOT / "inference" / "uv.lock"
HANDOFF = ROOT / "docs" / "windows-primary-handoff.md"


class WindowsSourceBringupScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.common = COMMON.read_text(encoding="utf-8")
        cls.provision = PROVISION.read_text(encoding="utf-8")
        cls.preflight = PREFLIGHT.read_text(encoding="utf-8")
        cls.run_source = RUN.read_text(encoding="utf-8")
        cls.job_smoke = JOB_SMOKE.read_text(encoding="utf-8")
        cls.build = BUILD.read_text(encoding="utf-8")
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")
        cls.handoff = HANDOFF.read_text(encoding="utf-8")

    def test_inference_lock_pins_the_qualification_client(self) -> None:
        document = tomllib.loads(INFERENCE_LOCK.read_text(encoding="utf-8"))
        packages = {
            package["name"]: package
            for package in document["package"]
        }
        self.assertEqual(packages["jarvis-inference"]["version"], "1.0.0")
        self.assertEqual(packages["openai"]["version"], "2.46.0")
        self.assertIn(
            {"name": "openai"},
            packages["jarvis-inference"]["optional-dependencies"]["qualification"],
        )

    def test_common_contract_requires_native_x64_python_and_pinned_uv(self) -> None:
        for fragment in (
            '$env:OS -ne "Windows_NT"',
            "& $PythonLauncher -3.11 -I -c",
            "-CommandType Application",
            "-LiteralPath $PythonLauncherPath",
            "-LiteralPath $UvExecutablePath",
            "UvExecutable = $UvExecutable",
            '$PythonIdentity.implementation -ne "CPython"',
            "$PythonIdentity.major -ne 3",
            "$PythonIdentity.minor -ne 11",
            "$PythonIdentity.bits -ne 64",
            '$script:CyrusRequiredUvVersion = "0.9.18"',
            '"^uv 0\\.9\\.18(?:\\s|$)"',
            '$env:UV_PYTHON_DOWNLOADS = "never"',
            '$env:HF_HUB_OFFLINE = "1"',
            '$env:TRANSFORMERS_OFFLINE = "1"',
            '$env:HF_DATASETS_OFFLINE = "1"',
            "function Assert-CyrusNoReparsePath",
            "function Test-CyrusLocalDriveAbsolutePath",
            'return (',
            '$LiteralPath -match "^[A-Za-z]:[\\\\/]"',
            '$Root -notmatch "^[A-Za-z]:[\\\\/]$"',
            "[IO.DriveType]::Network",
            'throw "windows_source.path_network_forbidden"',
            "[IO.FileAttributes]::ReparsePoint",
            "foreach ($Component in $Components)",
            "function New-CyrusSourceChildEnvironment",
            "function Set-CyrusProcessEnvironmentEntries",
            "function Restore-CyrusProcessEnvironmentSnapshot",
            "function Initialize-CyrusWindowsSourceNativeTypes",
            '[ValidateSet("Validator", "Inference", "Orchestrator", "Voice")]',
            "[Environment]::SystemDirectory",
            "[Environment]::GetFolderPath",
            "[string] $PrivateRuntimeDirectory",
            '$Values["PATH"] = [string]::Join(";", [string[]]$PathDirectories)',
            '$Values["TEMP"] = $RuntimeDirectory',
            '$Values["TMP"] = $RuntimeDirectory',
            '$Values["NO_PROXY"] = "127.0.0.1,localhost"',
            '$Values["JARVIS_INFERENCE_API_KEY"] = $InferenceBearer',
            "windows_source.inference_auth_scope_invalid",
        ):
            self.assertIn(fragment, self.common)

        self.assertNotIn("taskkill", self.common.lower())
        self.assertNotIn("Process.GetProcessById(", self.common)
        self.assertNotIn("[IO.Path]::IsPathFullyQualified", self.common)
        self.assertNotIn("[IO.Path]::IsPathFullyQualified", self.preflight)
        self.assertNotIn("[IO.Path]::IsPathFullyQualified", self.job_smoke)

        component_check = self.common.index("foreach ($Component in $Components)")
        canonical_resolve = self.common.index(
            "return (Resolve-Path -LiteralPath $FullPath -ErrorAction Stop).ProviderPath"
        )
        self.assertLess(component_check, canonical_resolve)

    def test_provision_uses_only_checked_locks_and_expected_windows_extras(self) -> None:
        for fragment in (
            "Assert-CyrusTrackedTreeClean -Repository $Repository -IncludeUntracked",
            '"lock", "--check", "--project", $Repository',
            '"lock", "--check", "--project", $InferenceDirectory',
            '"lock", "--check", "--project", $OrchestratorDirectory',
            '"sync"',
            '"--locked"',
            '"--no-editable"',
            '"--no-dev"',
            '"--extra", "asr"',
            '"--extra", "audio"',
            '"--extra", "vad"',
            '"--extra", "transport"',
            '"--extra", "windows"',
            '"--extra", "packaging"',
            '"--extra", "qualification"',
            '"--group", "dev"',
            'Join-Path $InferenceDirectory ".venv\\Scripts\\python.exe"',
            "m.version('openai') == '2.46.0'",
            "Assert-CyrusHandoffManifest",
            "model_download_performed = $false",
            "secret_created_or_modified = $false",
            "production_ready = $false",
            "Invoke-CyrusCheckedCommand $Toolchain.UvExecutable",
        ):
            self.assertIn(fragment, self.provision)

        forbidden = (
            "Invoke-WebRequest",
            "Invoke-RestMethod",
            "curl ",
            "wget ",
            "huggingface-cli",
            "hf download",
            "git clone",
            "pip install",
            "create-auth-key",
        )
        self.assertFalse(any(token in self.provision for token in forbidden))
        self.assertNotRegex(self.provision, r'(?m)^\s*uv\s+lock\s+(?!.*--check)')

    def test_preflight_reuses_strict_component_validators(self) -> None:
        for fragment in (
            '"validate"',
            '"jarvis_voice", "doctor"',
            '"jarvis_inference", "verify-manifest"',
            '"jarvis_inference", "verify-readiness"',
            "windows_source.inference_qualification_not_ready",
            "m.version('jarvis-inference') == '1.0.0'",
            "m.version('openai') == '2.46.0'",
            'Join-Path $Repository "inference\\.venv\\Scripts\\python.exe"',
            "windows_source.inference_auth_invalid",
            "$ConfiguredInferenceBearer.Length -gt 4096",
            "Set-CyrusProcessEnvironmentEntries -EnvironmentEntries $ValidatorEnvironment",
            '"cyrus-preflight-runtime-" + [Guid]::NewGuid().ToString("N")',
            '-GitExecutable $ResolvedTools["git"]',
            '-Role Validator',
            '-AdditionalPathDirectories $ToolDirectories',
            "inference_bearer_validated = $true",
            '$VoiceDocument.orchestrator_url -ne "ws://127.0.0.1:8765"',
            '$OrchestratorDocument.control_port -ne 8767',
            '$OrchestratorDocument.inference_base_url -ne "http://127.0.0.1:8766/v1"',
            "windows_source.fixed_endpoints_invalid",
            "Assert-CyrusPathOutsideRepository",
            "Assert-CyrusNoReparsePath",
            "-RequireContainer | Out-Null",
            "-RequireLeaf | Out-Null",
            '$VoiceDocument.windows_tts_backend -ne "sapi5"',
            '$VoiceDocument.scheduler.enabled -ne $false',
            '$OrchestratorDocument.memory_enabled -ne $false',
            '$OrchestratorDocument.memory_shadow_enabled -ne $false',
            '$OrchestratorDocument.research_enabled -ne $false',
            '$OrchestratorDocument.cloud_enabled -ne $false',
            '$OrchestratorDocument.tools_enabled -ne $false',
            "$OriginalProcessEnvironment",
            '"JARVIS_QUAL_COMPUTE_CAPABILITY"',
            "production_ready = $false",
            "network_used = $false",
        ):
            self.assertIn(fragment, self.preflight)
        self.assertNotIn("--active-cuda-probe", self.preflight)
        self.assertNotIn("create-auth-key", self.preflight)

    def test_preflight_freezes_cpu_and_cuda_turn_scoped_profiles(self) -> None:
        for fragment in (
            '[ValidateSet("CpuBaseline", "CudaTurnScoped")]',
            '$VoiceDocument.asr.model_profile -ne "base"',
            '$VoiceDocument.asr.execution_profile -ne "cpu"',
            '$VoiceDocument.asr.model_profile -ne "large-v3-turbo"',
            '$VoiceDocument.asr.execution_profile -ne "cuda"',
            '$VoiceDocument.asr.cuda.compute_type -ne "int8_float16"',
            '$VoiceDocument.asr.cuda.residency -ne "turn_scoped"',
            "$VoiceDocument.asr.cuda.minimum_free_vram_mb -ne 2048",
            "$VoiceDocument.asr.cuda.reserve_for_other_workloads_mb -ne 2048",
            '$VoiceDocument.asr.cuda.fallback_to_cpu -ne $true',
        ):
            self.assertIn(fragment, self.preflight)
        self.assertNotIn('residency -ne "warm"', self.preflight)

    def test_bringup_order_is_inference_then_orchestrator_then_voice(self) -> None:
        inference = self.run_source.index("$InferenceProcess = Start-CyrusSourceJobProcess")
        orchestrator = self.run_source.index("$OrchestratorProcess = Start-CyrusSourceJobProcess")
        voice = self.run_source.index("$VoiceProcess = Start-CyrusSourceJobProcess")
        status_definition = self.run_source.index("function Wait-CyrusStackReady")
        status_call = self.run_source.index(
            "Wait-CyrusStackReady -Processes $OwnedProcesses.ToArray()"
        )
        self.assertLess(inference, orchestrator)
        self.assertLess(orchestrator, voice)
        self.assertLess(status_definition, inference)
        self.assertLess(voice, status_call)

        for fragment in (
            "preflight_windows_source.ps1",
            '"http://127.0.0.1:8766/readyz"',
            '"http://127.0.0.1:8767/readyz"',
            '"-I", "-m", "jarvis_inference", "serve"',
            "$InferenceProcess = Start-CyrusSourceJobProcess",
            "-Executable $InferencePython",
            "$ConfiguredInferenceBearer",
            "$OriginalProcessEnvironment",
            "Set-CyrusProcessEnvironmentEntries -EnvironmentEntries $ValidatorEnvironment",
            '-PythonLauncherPath $ResolvedTools["py"]',
            '-UvExecutablePath $ResolvedTools["uv"]',
            "New-CyrusSourceChildEnvironment",
            "-Role Inference",
            "-Role Orchestrator",
            "-Role Voice",
            "-PrivateRuntimeDirectory $PrivateRuntimeDirectory",
            '"cyrus-source-runtime-" + [Guid]::NewGuid().ToString("N")',
            "windows_source.runtime_directory_cleanup_failed",
            "-not $HandleCloseFailed -and",
            "-InferenceBearer $ConfiguredInferenceBearer",
            "-EnvironmentEntries $InferenceEnvironment",
            "-EnvironmentEntries $OrchestratorEnvironment",
            "-EnvironmentEntries $VoiceEnvironment",
            "inference_bearer_not_exported = $true",
            "[switch] $PreflightOnly",
            '"jarvis_orchestrator.stack_cli"',
            '"e2e", "--turns", "1"',
            "full_local_stack_connected = $true",
            "production_ready = $false",
            "native_qualification_complete = $false",
            "autostart_installed = $false",
            "finally",
            "Stop-CyrusOwnedProcess",
            "windows_source.orphan_process_detected",
        ):
            self.assertIn(fragment, self.run_source)

    def test_source_children_are_assigned_suspended_to_one_kill_on_close_job(self) -> None:
        for fragment in (
            "CreateJobObjectW",
            "SetInformationJobObject",
            "JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000",
            "CreateProcessW",
            "CREATE_SUSPENDED = 0x00000004",
            "CREATE_UNICODE_ENVIRONMENT = 0x00000400",
            "AssignProcessToJobObject",
            "ResumeThread",
            "TerminateProcess",
            "TerminateJobObject",
            "WaitForSingleObject",
            "GetExitCodeProcess",
            "public sealed class OwnedProcess",
            "public OwnedProcess Start(",
            "Marshal.StringToHGlobalUni(environmentBlock)",
            "Marshal.FreeHGlobal(environmentPointer)",
            "Marshal.WriteInt16(environmentPointer, index * 2, 0)",
            "BuildEnvironmentBlock(environmentEntries)",
            "QuoteArgument(executable)",
            "commandLine.Length >= 32767",
            "if (created.hThread != IntPtr.Zero)",
            "if (created.hProcess != IntPtr.Zero)",
            "GC.SuppressFinalize(this)",
            "~ProcessJob()",
            "~OwnedProcess()",
            "function New-CyrusSourceProcessJob",
            "function Start-CyrusSourceJobProcess",
            "Assert-CyrusNoReparsePath",
        ):
            self.assertIn(fragment, self.common)

        created = self.common.index("if (!CreateProcessW(")
        assigned = self.common.index(
            "if (!AssignProcessToJobObject(jobHandle, created.hProcess))"
        )
        resumed = self.common.index("if (ResumeThread(created.hThread)")
        self.assertLess(created, assigned)
        self.assertLess(assigned, resumed)
        self.assertIn(
            "CREATE_SUSPENDED | CREATE_UNICODE_ENVIRONMENT", self.common
        )
        self.assertNotIn("IntPtr.Zero,\n                        workingDirectory", self.common)
        self.assertNotIn("Process.GetProcessById(", self.common)
        self.assertNotIn("taskkill", self.common.lower())

        self.assertNotRegex(self.run_source, r"(?m)^\s*Start-Process\b")
        self.assertNotIn("Process.Start", self.common)
        self.assertEqual(
            self.run_source.count("-Job $SourceProcessJob"),
            3,
        )
        for executable in (
            "$InferencePython",
            "$OrchestratorPython",
            "$VoicePython",
        ):
            self.assertIn(f"-Executable {executable}", self.run_source)

        dispose = self.run_source.index("$SourceProcessJob.Dispose()")
        fallback = self.run_source.index(
            "Stop-CyrusOwnedProcess $OwnedProcesses[$Index]"
        )
        parent_check = self.run_source.index(
            "foreach ($Process in $OwnedProcesses)", fallback
        )
        self.assertLess(dispose, fallback)
        self.assertLess(fallback, parent_check)

    def test_native_job_smoke_kills_a_child_after_its_parent_exits(self) -> None:
        for fragment in (
            "$Parent.Terminate()",
            "$Parent.WaitForExit(5000)",
            "$Stop = [DateTime]::UtcNow.AddSeconds(15)",
            "if ($Parent.HasExited)",
            "windows_source.job_smoke_child_exited_early",
            "$Job.Dispose()",
            "$Child.WaitForExit(10000)",
            "windows_source.job_smoke_orphan_survived",
            "windows_source.path_compatibility_false_accept",
            "windows_source.path_compatibility_false_reject",
            "powershell_51_path_contract = $true",
            "explicit_unicode_environment = $true",
            "ambient_secret_environment_stripped = $true",
            "validator_environment_stripped = $true",
            "inference_bearer_scoped = $true",
            "authoritative_process_handle = $true",
            "parent_exited_before_job_close = $true",
            "orphan_child_killed_on_job_close = $true",
        ):
            self.assertIn(fragment, self.job_smoke)
        self.assertLess(
            self.job_smoke.index("$Parent.Terminate()"),
            self.job_smoke.index("$Job.Dispose()"),
        )
        self.assertIn('"PING.EXE"', self.job_smoke)
        self.assertIn("Process.Start(start)", self.job_smoke)
        self.assertIn("start.CreateNoWindow = true", self.job_smoke)
        self.assertIn("-OutputType ConsoleApplication", self.job_smoke)
        self.assertNotIn("Start-Process", self.job_smoke)
        self.assertIn("CYRUS_UNICODE_PROBE=Κύρος", self.job_smoke)
        self.assertIn("must-not-be-inherited", self.job_smoke)
        self.assertIn(
            "windows_source.job_smoke_validator_environment_failed",
            self.job_smoke,
        )
        self.assertIn(
            "& .\\scripts\\test_windows_source_job.ps1",
            self.workflow,
        )
        self.assertIn("shell: powershell", self.workflow)
        self.assertIn("Expected Windows PowerShell 5.1.", self.workflow)
        self.assertNotIn("CYRUS_NATIVE_RELEASE_HOST_TEST", self.workflow)

    def test_bringup_does_not_install_or_claim_service_or_production(self) -> None:
        combined = self.preflight + self.run_source
        for forbidden in (
            "background install",
            "Register-ScheduledTask",
            "New-Service",
            "Start-Service",
            "sc.exe",
            "create-auth-key",
            "--active-cuda-probe",
            "Invoke-WebRequest",
            "huggingface",
        ):
            self.assertNotIn(forbidden, combined)
        self.assertNotRegex(
            combined,
            re.compile(r"production_ready\s*=\s*\$true", re.IGNORECASE),
        )

    def test_native_build_uses_only_provisioned_environments(self) -> None:
        for fragment in (
            '.venv\\Scripts\\python.exe',
            'inference\\.venv\\Scripts\\python.exe',
            'orchestrator\\.venv\\Scripts\\python.exe',
            "& $VoicePython -I -m PyInstaller",
            "windows_native_fixture_runner.spec",
            "native-fixture-runner.exe",
            "windows_asr_collector.spec",
            "cyrus-windows-asr-collector.exe",
            "1.2.1:4.8.1:6.21.0",
            "& $InferencePython -m unittest discover -s inference/tests -v",
            "& $InferencePython .\\inference\\tests\\sdk_conformance.py -v",
            "& $OrchestratorPython -m unittest discover -s orchestrator/tests -v",
            "m.version('openai')",
            '[ValidateSet("QualificationRunners", "VoicePackage")]',
            "CYRUS_WINDOWS_CUDA_BUNDLE_MANIFEST",
            "verify-cuda --manifest",
            '"boot-smoke"',
            "seal-runners",
            "verify-runners",
            "RunnerOutputDirectory must be a new directory",
            "Assert-CyrusTrackedTreeClean -Repository $Repository -IncludeUntracked",
            "Assert-CyrusPathOutsideRepository",
            "Assert-BootstrapNoReparsePath",
        ):
            self.assertIn(fragment, self.build)
        self.assertNotIn("py -3.11 -m", self.build)
        self.assertNotIn("$env:PYTHONPATH", self.build)
        self.assertNotIn("$env:CUDA_PATH", self.build)

    def test_handoff_documents_all_three_operator_commands_and_boundary(self) -> None:
        for fragment in (
            "scripts\\provision_windows_source.ps1",
            "preflight_windows_source.ps1",
            "scripts\\run_windows_source_beta.ps1",
            "inference\\.venv",
            "openai==2.46.0",
            "CpuBaseline",
            "CudaTurnScoped",
            "source-run",
            "not production",
        ):
            self.assertIn(fragment, self.handoff)


if __name__ == "__main__":
    unittest.main()
