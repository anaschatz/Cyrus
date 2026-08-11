"""Isolated GreekTTS-1.5 qualification harness for Apple Silicon.

The production TTS backend is intentionally untouched.  This module pins and
validates the upstream artifacts, converts the PEFT LoRA weights to MLX's native
adapter layout, and runs one offline Greek pronunciation fixture.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import resource
import shutil
import socket
import struct
import sys
import tempfile
import time
import wave
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Callable, Iterator, NamedTuple

from ..errors import BackendError


MLX_VERSION = "0.32.0"
MLX_LM_VERSION = "0.31.3"
TORCH_VERSION = "2.6.0"
SNAC_VERSION = "1.2.1"

BASE_REPO_ID = "mlx-community/orpheus-3b-0.1-ft-4bit"
BASE_REVISION = "b30aed4ecfa03c0177a5419b51919d14a105f183"
BASE_8BIT_REPO_ID = "mlx-community/orpheus-3b-0.1-ft-8bit"
BASE_8BIT_REVISION = "c8521bf4ea42f8dd96df099219d7871fc9873142"
ADAPTER_REPO_ID = "moiralabs/GreekTTS-1.5"
ADAPTER_REVISION = "0da4d9fff8b974fb1127f3c829f793dc0439dfd1"
SNAC_REPO_ID = "hubertsiuzdak/snac_24khz"
SNAC_REVISION = "d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec"
MLX_LM_PEFT_REFERENCE_COMMIT = "a224d34"

BASE_DIRECTORY = "base"
BASE_8BIT_DIRECTORY = "base-8bit"
PEFT_DIRECTORY = "peft"
SNAC_DIRECTORY = "snac"
CONVERTED_DIRECTORY = "mlx-adapter"
PEFT_CHECKPOINT_DIRECTORY = "checkpoint-264000"

GIB = 1024 * 1024 * 1024
MINIMUM_ACQUISITION_FREE_BYTES = 7 * GIB
MINIMUM_RUN_FREE_BYTES = 2 * GIB

TOKENIZER_LENGTH = 128_256
START_OF_SPEECH = TOKENIZER_LENGTH + 1
END_OF_SPEECH = TOKENIZER_LENGTH + 2
START_OF_HUMAN = TOKENIZER_LENGTH + 3
END_OF_HUMAN = TOKENIZER_LENGTH + 4
END_OF_TEXT = 128_009
AUDIO_TOKEN_OFFSET = 128_266
CODEBOOK_SIZE = 4_096
SAMPLE_RATE = 24_000
MODEL_LAYERS = 28
LORA_RANK = 64

PILOT_FIXTURE_ID = "greek_consonants"
PILOT_FIXTURE_TEXT = "Ο μπαμπάς πήγε στην αντάρα για να βρει τα αγκάθια."
EXPERIMENTAL_ORPHEUS_SPEAKERS = ("tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe")


class ArtifactSpec(NamedTuple):
    relative_path: str
    size: int
    sha256: str


BASE_ARTIFACTS = (
    ArtifactSpec("config.json", 1_068, "afb9eae45fd31a126d2aa13a6b24becb24113100cd5670bf32ab828eebc3ede7"),
    ArtifactSpec("model.safetensors", 1_857_062_388, "973488e7f7d681ac4f29dfb74d13ce01e94bf3aac201901a8a231d54d33ec7ed"),
    ArtifactSpec("model.safetensors.index.json", 45_720, "3cd0aefd02957e8302d260c9b2a722b4c8662c8a24e710d43ef7399b45038e37"),
    ArtifactSpec("special_tokens_map.json", 350, "02726c1fa32b9ca7e0630f81bf34b76998fe9353a43089d571ee826bae672b25"),
    ArtifactSpec("tokenizer.json", 22_849_547, "fc3fecb199b4170636dbfab986d25f628157268d37b861f9cadaca60b1353bce"),
    ArtifactSpec("tokenizer_config.json", 5_407_395, "cc63c5a97dd0daef0ab805a44e440464aa82ffa8923e6a6477a330b1f96de339"),
)
BASE_8BIT_ARTIFACTS = (
    ArtifactSpec("config.json", 1_068, "66d8d06fb6c67c507d8993d1b2c8d9dce34002734bdb4c7d4990473e0a33aaeb"),
    ArtifactSpec("model.safetensors", 3_507_408_624, "28d2b2711abc8f7585e91ee44f91b69c7eaa37ee716af1c6ee421cc666628def"),
    ArtifactSpec("model.safetensors.index.json", 45_720, "5810473b747abf0d82e9ac222304d672a2aeae7b13137a6314d1e2e9cfe92e2a"),
    ArtifactSpec("special_tokens_map.json", 350, "02726c1fa32b9ca7e0630f81bf34b76998fe9353a43089d571ee826bae672b25"),
    ArtifactSpec("tokenizer.json", 22_849_547, "fc3fecb199b4170636dbfab986d25f628157268d37b861f9cadaca60b1353bce"),
    ArtifactSpec("tokenizer_config.json", 5_407_395, "cc63c5a97dd0daef0ab805a44e440464aa82ffa8923e6a6477a330b1f96de339"),
)
PEFT_ARTIFACTS = (
    ArtifactSpec(
        f"{PEFT_CHECKPOINT_DIRECTORY}/adapter_config.json",
        1_071,
        "2178e876b1bcdb3f7fc6dacef36230edca904b9e76a00fce35756b75358ba504",
    ),
    ArtifactSpec(
        f"{PEFT_CHECKPOINT_DIRECTORY}/adapter_model.safetensors",
        389_074_464,
        "37bce0e22252d3d9d00cee24db4bec5940419deb0334e3781e01322513c804e6",
    ),
)
SNAC_ARTIFACTS = (
    ArtifactSpec("config.json", 300, "e119b9366d4f5e73c6ca5f31137c4ff361578bbb132953a5203afe037c4012be"),
    ArtifactSpec("pytorch_model.bin", 79_488_254, "4b8164cc6606bfa627f1a784734c1e539891518f1191ed9194fe1e3b9b4bff40"),
)
MODEL_DOWNLOAD_BYTES = sum(item.size for item in BASE_ARTIFACTS + PEFT_ARTIFACTS + SNAC_ARTIFACTS)
MODEL_8BIT_DOWNLOAD_BYTES = sum(item.size for item in BASE_8BIT_ARTIFACTS + PEFT_ARTIFACTS + SNAC_ARTIFACTS)
CONVERSION_RESERVE_BYTES = PEFT_ARTIFACTS[-1].size + 512 * 1024 * 1024


class BaseVariantSpec(NamedTuple):
    variant: str
    directory: str
    repo_id: str
    revision: str
    quantization_bits: int
    artifacts: tuple[ArtifactSpec, ...]


BASE_VARIANTS = {
    "4bit": BaseVariantSpec("4bit", BASE_DIRECTORY, BASE_REPO_ID, BASE_REVISION, 4, BASE_ARTIFACTS),
    "8bit": BaseVariantSpec(
        "8bit",
        BASE_8BIT_DIRECTORY,
        BASE_8BIT_REPO_ID,
        BASE_8BIT_REVISION,
        8,
        BASE_8BIT_ARTIFACTS,
    ),
}

PEFT_TARGET_TO_MLX = {
    "q_proj": "self_attn.q_proj",
    "k_proj": "self_attn.k_proj",
    "v_proj": "self_attn.v_proj",
    "o_proj": "self_attn.o_proj",
    "gate_proj": "mlp.gate_proj",
    "up_proj": "mlp.up_proj",
    "down_proj": "mlp.down_proj",
}
EXPECTED_MLX_KEYS = tuple(PEFT_TARGET_TO_MLX.values())
_PEFT_WEIGHT_RE = re.compile(
    r"^base_model\.model\.model\.layers\.(?P<layer>\d+)\."
    r"(?P<section>self_attn|mlp)\.(?P<target>[a-z_]+)\."
    r"lora_(?P<matrix>A|B)\.weight$"
)


@dataclass(frozen=True, slots=True)
class ModelValidation:
    ready: bool
    code: str
    files_present: int
    files_required: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _exact_distribution_version(name: str, expected: str) -> bool:
    try:
        return importlib.metadata.version(name) == expected
    except importlib.metadata.PackageNotFoundError:
        return False


def _read_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _validate_artifacts(directory: Path, artifacts: tuple[ArtifactSpec, ...]) -> tuple[bool, int]:
    present = 0
    for artifact in artifacts:
        path = directory / artifact.relative_path
        try:
            valid = not path.is_symlink() and path.is_file() and path.stat().st_size == artifact.size
            if not valid or _sha256(path) != artifact.sha256:
                return False, present
        except OSError:
            return False, present
        present += 1
    return True, present


def _select_base_variant(base_variant: str) -> BaseVariantSpec:
    try:
        return BASE_VARIANTS[base_variant]
    except KeyError as exc:
        raise BackendError(
            "tts.greektts15.base_variant_invalid",
            "GreekTTS base variant is invalid",
            subsystem="tts",
        ) from exc


def _validate_runtime_schemas(model_root: Path, base: BaseVariantSpec) -> bool:
    base_config = _read_json(model_root / base.directory / "config.json")
    adapter = _read_json(
        model_root / PEFT_DIRECTORY / PEFT_CHECKPOINT_DIRECTORY / "adapter_config.json"
    )
    snac = _read_json(model_root / SNAC_DIRECTORY / "config.json")
    if base_config is None or adapter is None or snac is None:
        return False
    return (
        base_config.get("model_type") == "llama"
        and base_config.get("num_hidden_layers") == MODEL_LAYERS
        and base_config.get("hidden_size") == 3_072
        and base_config.get("intermediate_size") == 8_192
        and base_config.get("vocab_size") == 156_940
        and base_config.get("quantization") == {"group_size": 64, "bits": base.quantization_bits}
        and adapter.get("peft_type") == "LORA"
        and adapter.get("r") == LORA_RANK
        and adapter.get("lora_alpha") == LORA_RANK
        and adapter.get("lora_dropout") == 0
        and adapter.get("use_dora") is False
        and adapter.get("use_rslora") is False
        and set(adapter.get("target_modules", [])) == set(PEFT_TARGET_TO_MLX)
        and snac.get("sampling_rate") == SAMPLE_RATE
        and snac.get("codebook_size") == CODEBOOK_SIZE
        and snac.get("vq_strides") == [4, 2, 1]
    )


def validate_greektts_model_root(
    model_root: str | Path,
    *,
    base_variant: str = "4bit",
) -> ModelValidation:
    root = Path(model_root).expanduser()
    base = _select_base_variant(base_variant)
    artifacts = (
        (root / base.directory, base.artifacts),
        (root / PEFT_DIRECTORY, PEFT_ARTIFACTS),
        (root / SNAC_DIRECTORY, SNAC_ARTIFACTS),
    )
    required = sum(len(group) for _, group in artifacts)
    if not root.is_absolute() or not root.is_dir() or root.is_symlink():
        return ModelValidation(False, "tts.greektts15.model_root_missing", 0, required)
    present = 0
    for directory, specs in artifacts:
        valid, group_present = _validate_artifacts(directory, specs)
        present += group_present
        if not valid:
            return ModelValidation(False, "tts.greektts15.model_artifact_invalid", present, required)
    if not _validate_runtime_schemas(root, base):
        return ModelValidation(False, "tts.greektts15.model_schema_invalid", present, required)
    return ModelValidation(True, "tts.greektts15.model_ready", required, required)


def _native_adapter_config() -> dict[str, object]:
    return {
        "fine_tune_type": "lora",
        "num_layers": MODEL_LAYERS,
        "lora_parameters": {
            "rank": LORA_RANK,
            "scale": 1.0,
            "dropout": 0.0,
            "keys": list(EXPECTED_MLX_KEYS),
        },
    }


def validate_converted_adapter(model_root: str | Path) -> ModelValidation:
    root = Path(model_root).expanduser()
    directory = root / CONVERTED_DIRECTORY
    config_path = directory / "adapter_config.json"
    weights_path = directory / "adapters.safetensors"
    manifest_path = directory / "conversion_manifest.json"
    required = 3
    if not directory.is_dir() or directory.is_symlink():
        return ModelValidation(False, "tts.greektts15.adapter_not_converted", 0, required)
    for path in (config_path, weights_path, manifest_path):
        if path.is_symlink() or not path.is_file():
            return ModelValidation(False, "tts.greektts15.converted_adapter_invalid", 0, required)
    config = _read_json(config_path)
    manifest = _read_json(manifest_path)
    if config != _native_adapter_config() or manifest is None:
        return ModelValidation(False, "tts.greektts15.converted_adapter_invalid", 0, required)
    expected_tensors = MODEL_LAYERS * len(EXPECTED_MLX_KEYS) * 2
    try:
        valid = (
            manifest.get("schema_version") == 1
            and manifest.get("source_revision") == ADAPTER_REVISION
            and manifest.get("source_sha256") == PEFT_ARTIFACTS[-1].sha256
            and manifest.get("mlx_lm_reference_commit") == MLX_LM_PEFT_REFERENCE_COMMIT
            and manifest.get("weight_tensors") == expected_tensors
            and manifest.get("output_size") == weights_path.stat().st_size
            and manifest.get("output_sha256") == _sha256(weights_path)
        )
    except OSError:
        valid = False
    if not valid:
        return ModelValidation(False, "tts.greektts15.converted_adapter_invalid", 0, required)
    return ModelValidation(True, "tts.greektts15.adapter_converted", required, required)


def _dependency_report(
    probe: Callable[[str, str], bool] = _exact_distribution_version,
) -> dict[str, bool]:
    return {
        "mlx": bool(probe("mlx", MLX_VERSION)),
        "mlx_lm": bool(probe("mlx-lm", MLX_LM_VERSION)),
        "torch": bool(probe("torch", TORCH_VERSION)),
        "snac": bool(probe("snac", SNAC_VERSION)),
    }


def greektts15_pilot_preflight(
    storage_root: str | Path,
    model_root: str | Path,
    *,
    platform_name: str | None = None,
    machine: str | None = None,
    python_version: tuple[int, int, int] | None = None,
    pointer_bits: int | None = None,
    disk_usage: Callable[[str | os.PathLike[str]], object] = shutil.disk_usage,
    dependency_probe: Callable[[str, str], bool] = _exact_distribution_version,
    base_variant: str = "4bit",
) -> dict[str, object]:
    storage = Path(storage_root).expanduser()
    if not storage.is_absolute() or not storage.is_dir() or storage.is_symlink():
        raise BackendError(
            "tts.greektts15.storage_root_invalid",
            "GreekTTS pilot storage root is invalid",
            subsystem="tts",
        )
    selected_platform = sys.platform if platform_name is None else platform_name
    selected_machine = platform.machine() if machine is None else machine
    selected_version = sys.version_info[:3] if python_version is None else python_version
    selected_bits = struct.calcsize("P") * 8 if pointer_bits is None else pointer_bits
    try:
        available = int(getattr(disk_usage(storage), "free"))
    except (OSError, TypeError, ValueError, AttributeError) as exc:
        raise BackendError(
            "tts.greektts15.disk_probe_failed",
            "GreekTTS pilot disk probe failed",
            subsystem="tts",
        ) from exc

    host_ready = (
        selected_platform == "darwin"
        and selected_machine == "arm64"
        and selected_version[:2] == (3, 11)
        and selected_bits == 64
    )
    dependencies = _dependency_report(dependency_probe)
    dependencies_ready = all(dependencies.values())
    base = _select_base_variant(base_variant)
    upstream = validate_greektts_model_root(model_root, base_variant=base.variant)
    converted = validate_converted_adapter(model_root)
    model_download_bytes = sum(item.size for item in base.artifacts + PEFT_ARTIFACTS + SNAC_ARTIFACTS)
    exact_acquisition = model_download_bytes + CONVERSION_RESERVE_BYTES + MINIMUM_RUN_FREE_BYTES
    acquisition_required = exact_acquisition if dependencies_ready else MINIMUM_ACQUISITION_FREE_BYTES
    acquisition_ready = host_ready and available >= acquisition_required
    run_ready = (
        host_ready
        and available >= MINIMUM_RUN_FREE_BYTES
        and dependencies_ready
        and upstream.ready
        and converted.ready
    )

    if not host_ready:
        code = "tts.greektts15.host_unsupported"
    elif available < MINIMUM_RUN_FREE_BYTES:
        code = "tts.greektts15.disk_space_critical"
    elif run_ready:
        code = "tts.greektts15.pilot_ready"
    elif available < acquisition_required:
        code = "tts.greektts15.disk_space_insufficient"
    elif not dependencies_ready:
        code = "tts.greektts15.dependencies_missing"
    elif not upstream.ready:
        code = upstream.code
    else:
        code = converted.code

    return {
        "schema_version": 1,
        "ready": run_ready,
        "code": code,
        "acquisition_ready": acquisition_ready,
        "host": {
            "platform": selected_platform,
            "machine": selected_machine,
            "python": ".".join(str(item) for item in selected_version),
            "bits": selected_bits,
        },
        "disk": {
            "available_mib": available // (1024 * 1024),
            "acquisition_required_mib": acquisition_required // (1024 * 1024),
            "run_required_mib": MINIMUM_RUN_FREE_BYTES // (1024 * 1024),
            "model_download_mib": model_download_bytes // (1024 * 1024),
            "conversion_reserve_mib": CONVERSION_RESERVE_BYTES // (1024 * 1024),
        },
        "dependencies": dependencies,
        "upstream": {
            "ready": upstream.ready,
            "code": upstream.code,
            "files_present": upstream.files_present,
            "files_required": upstream.files_required,
        },
        "adapter_conversion": {
            "ready": converted.ready,
            "code": converted.code,
            "reference_commit": MLX_LM_PEFT_REFERENCE_COMMIT,
        },
        "models": {
            "base": {
                "variant": base.variant,
                "repo_id": base.repo_id,
                "revision": base.revision,
                "quantization_bits": base.quantization_bits,
            },
            "adapter": {"repo_id": ADAPTER_REPO_ID, "revision": ADAPTER_REVISION},
            "snac": {"repo_id": SNAC_REPO_ID, "revision": SNAC_REVISION},
        },
        "network_used": False,
        "installation_performed": False,
        "runtime_started": False,
        "content_captured": False,
    }


def _prepare_private_model_root(storage_root: Path, model_root: Path) -> None:
    try:
        resolved_storage = storage_root.resolve(strict=True)
        resolved_parent = model_root.parent.resolve(strict=True)
    except OSError as exc:
        raise BackendError(
            "tts.greektts15.model_root_invalid",
            "GreekTTS model root is invalid",
            subsystem="tts",
        ) from exc
    if (
        not model_root.is_absolute()
        or model_root.is_symlink()
        or resolved_parent != resolved_storage
    ):
        raise BackendError(
            "tts.greektts15.model_root_invalid",
            "GreekTTS model root must be a direct private storage child",
            subsystem="tts",
        )
    model_root.mkdir(mode=0o700, exist_ok=True)
    os.chmod(model_root, 0o700)


def acquire_greektts15_models(
    storage_root: str | Path,
    model_root: str | Path,
    *,
    base_variant: str = "4bit",
) -> dict[str, object]:
    storage = Path(storage_root).expanduser()
    root = Path(model_root).expanduser()
    base = _select_base_variant(base_variant)
    report = greektts15_pilot_preflight(storage, root, base_variant=base.variant)
    if not report["acquisition_ready"]:
        raise BackendError(str(report["code"]), "GreekTTS acquisition preflight failed", subsystem="tts")
    _prepare_private_model_root(storage, root)
    validation = validate_greektts_model_root(root, base_variant=base.variant)
    if validation.ready:
        return {
            "schema_version": 1,
            "ready": True,
            "code": "tts.greektts15.models_already_ready",
            "network_used": False,
            "downloaded_bytes": 0,
        }
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BackendError(
            "tts.greektts15.dependencies_missing",
            "Hugging Face Hub is unavailable",
            subsystem="tts",
        ) from exc

    downloads = (
        (base.repo_id, base.revision, base.artifacts, root / base.directory),
        (ADAPTER_REPO_ID, ADAPTER_REVISION, PEFT_ARTIFACTS, root / PEFT_DIRECTORY),
        (SNAC_REPO_ID, SNAC_REVISION, SNAC_ARTIFACTS, root / SNAC_DIRECTORY),
    )
    for repo_id, revision, artifacts, directory in downloads:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            allow_patterns=[item.relative_path for item in artifacts],
            local_dir=directory,
            max_workers=1,
        )
    validation = validate_greektts_model_root(root, base_variant=base.variant)
    if not validation.ready:
        raise BackendError(validation.code, "Downloaded GreekTTS models failed validation", subsystem="tts")
    for directory, artifacts in (
        (root / base.directory, base.artifacts),
        (root / PEFT_DIRECTORY, PEFT_ARTIFACTS),
        (root / SNAC_DIRECTORY, SNAC_ARTIFACTS),
    ):
        os.chmod(directory, 0o700)
        for artifact in artifacts:
            os.chmod(directory / artifact.relative_path, 0o600)
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.greektts15.models_acquired",
        "network_used": True,
        "downloaded_bytes": sum(item.size for item in base.artifacts + PEFT_ARTIFACTS + SNAC_ARTIFACTS),
        "base_variant": base.variant,
    }


def remap_peft_weight_key(key: str) -> str | None:
    """Map one standard PEFT Llama LoRA key to MLX's adapter parameter name."""

    match = _PEFT_WEIGHT_RE.fullmatch(key)
    if match is None:
        return None
    layer = int(match.group("layer"))
    target = match.group("target")
    section = match.group("section")
    expected = PEFT_TARGET_TO_MLX.get(target)
    if layer >= MODEL_LAYERS or expected is None or not expected.startswith(f"{section}."):
        return None
    matrix = "a" if match.group("matrix") == "A" else "b"
    return f"model.layers.{layer}.{expected}.lora_{matrix}"


def expected_converted_weight_keys() -> set[str]:
    return {
        f"model.layers.{layer}.{target}.lora_{matrix}"
        for layer in range(MODEL_LAYERS)
        for target in EXPECTED_MLX_KEYS
        for matrix in ("a", "b")
    }


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def convert_greektts15_adapter(model_root: str | Path) -> dict[str, object]:
    root = Path(model_root).expanduser()
    validation = validate_greektts_model_root(root)
    if not validation.ready:
        raise BackendError(validation.code, "GreekTTS source adapter is unavailable", subsystem="tts")
    converted = validate_converted_adapter(root)
    if converted.ready:
        return {"schema_version": 1, "ready": True, "code": "tts.greektts15.adapter_already_converted"}
    try:
        import mlx.core as mx
    except ImportError as exc:
        raise BackendError("tts.greektts15.dependencies_missing", "MLX is unavailable", subsystem="tts") from exc

    source = root / PEFT_DIRECTORY / PEFT_CHECKPOINT_DIRECTORY / "adapter_model.safetensors"
    destination = root / CONVERTED_DIRECTORY
    if destination.is_symlink():
        raise BackendError("tts.greektts15.converted_adapter_invalid", "Converted adapter path is unsafe", subsystem="tts")
    destination.mkdir(mode=0o700, exist_ok=True)
    os.chmod(destination, 0o700)
    raw_weights = mx.load(str(source))
    remapped: dict[str, object] = {}
    for key, value in raw_weights.items():
        mapped = remap_peft_weight_key(key)
        if mapped is None:
            if ".lora_" in key:
                raise BackendError(
                    "tts.greektts15.adapter_key_unsupported",
                    "GreekTTS adapter contains an unsupported LoRA key",
                    subsystem="tts",
                )
            continue
        if mapped in remapped:
            raise BackendError("tts.greektts15.adapter_key_duplicate", "Duplicate LoRA key", subsystem="tts")
        remapped[mapped] = mx.transpose(value)
    expected = expected_converted_weight_keys()
    if set(remapped) != expected:
        raise BackendError(
            "tts.greektts15.adapter_key_set_invalid",
            "GreekTTS adapter does not contain the complete expected LoRA key set",
            subsystem="tts",
        )

    descriptor, temporary_name = tempfile.mkstemp(prefix=".adapters.", suffix=".safetensors", dir=destination)
    os.close(descriptor)
    output_path = destination / "adapters.safetensors"
    try:
        mx.save_safetensors(temporary_name, remapped, metadata={"format": "mlx"})
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, output_path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
    _atomic_json(destination / "adapter_config.json", _native_adapter_config())
    manifest = {
        "schema_version": 1,
        "source_revision": ADAPTER_REVISION,
        "source_sha256": PEFT_ARTIFACTS[-1].sha256,
        "mlx_lm_reference_commit": MLX_LM_PEFT_REFERENCE_COMMIT,
        "weight_tensors": len(remapped),
        "output_size": output_path.stat().st_size,
        "output_sha256": _sha256(output_path),
    }
    _atomic_json(destination / "conversion_manifest.json", manifest)
    converted = validate_converted_adapter(root)
    if not converted.ready:
        raise BackendError(converted.code, "Converted GreekTTS adapter failed validation", subsystem="tts")
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.greektts15.adapter_converted",
        "weight_tensors": len(remapped),
        "output_size": manifest["output_size"],
        "output_sha256": manifest["output_sha256"],
    }


def extract_snac_codes(generated_tokens: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Extract and validate the three SNAC code layers from Orpheus output."""

    starts = [index for index, token in enumerate(generated_tokens) if token == START_OF_SPEECH]
    if not starts:
        raise BackendError("tts.greektts15.start_of_speech_missing", "No speech token was generated", subsystem="tts")
    audio_tokens = [token for token in generated_tokens[starts[-1] + 1 :] if token != END_OF_SPEECH]
    audio_tokens = audio_tokens[: (len(audio_tokens) // 7) * 7]
    if not audio_tokens:
        raise BackendError("tts.greektts15.audio_tokens_missing", "No complete audio frame was generated", subsystem="tts")
    values = [token - AUDIO_TOKEN_OFFSET for token in audio_tokens]
    layer_1: list[int] = []
    layer_2: list[int] = []
    layer_3: list[int] = []
    for offset in range(0, len(values), 7):
        frame = values[offset : offset + 7]
        layer_1.append(frame[0])
        layer_2.extend((frame[1] - CODEBOOK_SIZE, frame[4] - 4 * CODEBOOK_SIZE))
        layer_3.extend(
            (
                frame[2] - 2 * CODEBOOK_SIZE,
                frame[3] - 3 * CODEBOOK_SIZE,
                frame[5] - 5 * CODEBOOK_SIZE,
                frame[6] - 6 * CODEBOOK_SIZE,
            )
        )
    if any(code < 0 or code >= CODEBOOK_SIZE for layer in (layer_1, layer_2, layer_3) for code in layer):
        raise BackendError("tts.greektts15.audio_code_invalid", "Generated SNAC code is out of range", subsystem="tts")
    return layer_1, layer_2, layer_3


def _deny_network_during_pilot(function: Callable[..., dict[str, object]]) -> Callable[..., dict[str, object]]:
    @wraps(function)
    def guarded(*args: object, **kwargs: object) -> dict[str, object]:
        original_socket = socket.socket
        original_create_connection = socket.create_connection

        class OfflineSocket(original_socket):
            def connect(self, address: object) -> None:
                del address
                raise BackendError(
                    "tts.greektts15.network_forbidden",
                    "Network access is forbidden during the GreekTTS pilot",
                    subsystem="tts",
                )

            def connect_ex(self, address: object) -> int:
                self.connect(address)
                return 1

        def deny_create_connection(*args: object, **kwargs: object) -> object:
            del args, kwargs
            raise BackendError(
                "tts.greektts15.network_forbidden",
                "Network access is forbidden during the GreekTTS pilot",
                subsystem="tts",
            )

        socket.socket = OfflineSocket
        socket.create_connection = deny_create_connection
        try:
            return function(*args, **kwargs)
        finally:
            socket.socket = original_socket
            socket.create_connection = original_create_connection

    return guarded


@contextmanager
def _offline_model_environment() -> Iterator[None]:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: "1" for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _write_pcm16_wav(path: Path, waveform: object, torch_module: object) -> tuple[int, float, float]:
    tensor = waveform.detach().squeeze().to("cpu", dtype=torch_module.float32)
    if tensor.ndim != 1 or tensor.numel() == 0 or not bool(torch_module.isfinite(tensor).all().item()):
        raise BackendError("tts.greektts15.audio_invalid", "SNAC produced invalid audio", subsystem="tts")
    peak_before = float(tensor.abs().max().item())
    gain = min(1.0, 0.95 / peak_before) if peak_before > 0.0 else 1.0
    tensor = (tensor * gain).clamp(-1.0, 1.0)
    pcm = (tensor * 32_767.0).round().to(torch_module.int16).numpy().tobytes()
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm)
    os.chmod(path, 0o600)
    return int(tensor.numel()), peak_before, gain


def format_pilot_prompt(speaker: str | None = None) -> str:
    """Format the fixed fixture with an optional, explicitly experimental base speaker."""

    if speaker is None:
        return PILOT_FIXTURE_TEXT
    if speaker not in EXPERIMENTAL_ORPHEUS_SPEAKERS:
        raise BackendError(
            "tts.greektts15.speaker_invalid",
            "GreekTTS experimental speaker is invalid",
            subsystem="tts",
        )
    return f"{speaker}: {PILOT_FIXTURE_TEXT}"


@_deny_network_during_pilot
def run_greektts15_pilot(
    model_root: str | Path,
    output_directory: str | Path,
    *,
    max_new_tokens: int = 1_200,
    seed: int = 10_001,
    base_variant: str = "4bit",
    speaker: str | None = None,
) -> dict[str, object]:
    root = Path(model_root).expanduser()
    base = _select_base_variant(base_variant)
    upstream = validate_greektts_model_root(root, base_variant=base.variant)
    converted = validate_converted_adapter(root)
    if not upstream.ready:
        raise BackendError(upstream.code, "GreekTTS model root is not ready", subsystem="tts")
    if not converted.ready:
        raise BackendError(converted.code, "GreekTTS adapter is not converted", subsystem="tts")
    if not all(_dependency_report().values()):
        raise BackendError("tts.greektts15.dependencies_missing", "GreekTTS dependencies are unavailable", subsystem="tts")
    output = Path(output_directory).expanduser()
    if not output.is_absolute() or output.is_symlink():
        raise BackendError("tts.greektts15.output_invalid", "GreekTTS output directory is invalid", subsystem="tts")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(output, 0o700)
    if max_new_tokens < 7 or max_new_tokens > 1_200:
        raise BackendError("tts.greektts15.max_tokens_invalid", "GreekTTS token limit is invalid", subsystem="tts")
    prompt_text = format_pilot_prompt(speaker)

    try:
        import mlx.core as mx
        import torch
        from mlx_lm import load
        from mlx_lm.generate import generate_step
        from mlx_lm.sample_utils import make_logits_processors, make_sampler
        from mlx_lm.tuner.lora import LoRALinear
        from snac import SNAC
    except ImportError as exc:  # pragma: no cover - isolated pilot only
        raise BackendError("tts.greektts15.dependencies_missing", "GreekTTS dependencies are unavailable", subsystem="tts") from exc

    with _offline_model_environment():
        mx.random.seed(seed)
        load_started = time.perf_counter()
        model, tokenizer = load(
            str(root / base.directory),
            adapter_path=str(root / CONVERTED_DIRECTORY),
            lazy=False,
        )
        mx.synchronize()
        model_load_ms = (time.perf_counter() - load_started) * 1000.0
        lora_modules = sum(isinstance(module, LoRALinear) for _, module in model.named_modules())
        expected_modules = MODEL_LAYERS * len(EXPECTED_MLX_KEYS)
        if lora_modules != expected_modules:
            raise BackendError("tts.greektts15.adapter_load_invalid", "MLX did not attach every LoRA module", subsystem="tts")

        snac_started = time.perf_counter()
        snac_model = SNAC.from_pretrained(str(root / SNAC_DIRECTORY)).to("cpu").eval()
        snac_load_ms = (time.perf_counter() - snac_started) * 1000.0
        text_ids = list(tokenizer.encode(prompt_text, add_special_tokens=True))
        if not text_ids or text_ids[0] != 128_000:
            raise BackendError("tts.greektts15.tokenizer_invalid", "Tokenizer did not emit start-of-text", subsystem="tts")
        prompt_ids = [START_OF_HUMAN, *text_ids, END_OF_TEXT, END_OF_HUMAN]
        sampler = make_sampler(temp=0.6, top_p=0.95)
        logits_processors = make_logits_processors(repetition_penalty=1.1)
        generated: list[int] = []
        first_speech_ms: float | None = None
        generation_started = time.perf_counter()
        for token, _ in generate_step(
            mx.array(prompt_ids),
            model,
            max_tokens=max_new_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            token_id = int(token if isinstance(token, int) else token.item())
            generated.append(token_id)
            if token_id == START_OF_SPEECH and first_speech_ms is None:
                first_speech_ms = (time.perf_counter() - generation_started) * 1000.0
            if token_id == END_OF_SPEECH:
                break
        mx.synchronize()
        generation_ms = (time.perf_counter() - generation_started) * 1000.0
        if not generated or generated[-1] != END_OF_SPEECH:
            raise BackendError("tts.greektts15.end_of_speech_missing", "Speech generation reached its token limit", subsystem="tts")
        code_layers = extract_snac_codes(generated)
        decode_started = time.perf_counter()
        with torch.inference_mode():
            codes = [torch.tensor(layer, dtype=torch.long).unsqueeze(0) for layer in code_layers]
            waveform = snac_model.decode(codes)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0
        suffix = "" if base.variant == "4bit" else f"-{base.variant}"
        if speaker is not None:
            suffix = f"{suffix}-{speaker}"
        destination = output / f"01-{PILOT_FIXTURE_ID}{suffix}.wav"
        sample_count, peak_before, gain = _write_pcm16_wav(destination, waveform, torch)

    duration_ms = sample_count * 1000.0 / SAMPLE_RATE
    total_synthesis_ms = generation_ms + decode_ms
    return {
        "schema_version": 1,
        "ready": True,
        "code": "tts.greektts15.pilot_completed",
        "backend": f"greektts_1_5_mlx_{base.variant}",
        "device": "mlx_apple_silicon",
        "language": "el",
        "fixture_id": PILOT_FIXTURE_ID,
        "models": {
            "base": {
                "variant": base.variant,
                "repo_id": base.repo_id,
                "revision": base.revision,
                "quantization_bits": base.quantization_bits,
            },
            "adapter": {"repo_id": ADAPTER_REPO_ID, "revision": ADAPTER_REVISION},
            "snac": {"repo_id": SNAC_REPO_ID, "revision": SNAC_REVISION},
        },
        "adapter": {"lora_modules": lora_modules, "scale": 1.0},
        "voice": {
            "mode": "experimental_foundation_prompt" if speaker is not None else "greek_adapter_default",
            "speaker": speaker,
        },
        "model_load_ms": round(model_load_ms, 3),
        "snac_load_ms": round(snac_load_ms, 3),
        "generation_ms": round(generation_ms, 3),
        "time_to_first_speech_token_ms": round(first_speech_ms, 3) if first_speech_ms is not None else None,
        "warm_audio_ready_ms": round(total_synthesis_ms, 3),
        "streaming_enabled": False,
        "decode_ms": round(decode_ms, 3),
        "audio_duration_ms": round(duration_ms, 3),
        "realtime_factor": round(total_synthesis_ms / duration_ms, 4) if duration_ms else None,
        "generated_tokens": len(generated),
        "audio_frames": len(code_layers[0]),
        "peak_before_normalization": round(peak_before, 6),
        "normalization_gain": round(gain, 6),
        "peak_mlx_memory_bytes": int(mx.get_peak_memory()),
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "sample": {"file": destination.name, "sample_rate": SAMPLE_RATE, "channels": 1},
        "network_used": False,
        "network_guard": "python_socket_fail_closed",
        "production_backend_changed": False,
        "content_captured": False,
    }
