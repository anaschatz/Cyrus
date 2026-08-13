"""Dependency-light acoustic diagnostics for GreekTTS listening candidates."""

from __future__ import annotations

from pathlib import Path

from jarvis_voice.errors import BackendError


def analyze_acoustic_file(audio_path: str | Path) -> dict[str, object]:
    """Measure discontinuities without claiming to replace human listening."""

    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise BackendError(
            "tts.greektts15.acoustic_dependencies_missing",
            "GreekTTS acoustic diagnostic dependencies are missing",
            subsystem="tts",
        ) from exc
    path = Path(audio_path).expanduser()
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BackendError(
            "tts.greektts15.acoustic_input_invalid",
            "GreekTTS acoustic input is invalid",
            subsystem="tts",
        )
    waveform, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if waveform.ndim != 1 or sample_rate < 16_000 or waveform.size < sample_rate // 2 or not np.isfinite(waveform).all():
        raise BackendError(
            "tts.greektts15.acoustic_input_invalid",
            "GreekTTS acoustic waveform is invalid",
            subsystem="tts",
        )
    frame_length = round(sample_rate * 0.04)
    hop = round(sample_rate * 0.01)
    frame_count = 1 + (waveform.size - frame_length) // hop
    frames = np.stack([waveform[index * hop : index * hop + frame_length] for index in range(frame_count)])
    windowed = frames * np.hanning(frame_length).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    active_threshold = max(-50.0, float(np.percentile(rms_db, 90)) - 32.0)
    active = rms_db >= active_threshold
    active_indices = np.flatnonzero(active)
    if active_indices.size < 4:
        raise BackendError(
            "tts.greektts15.acoustic_input_invalid",
            "GreekTTS acoustic waveform contains too little speech",
            subsystem="tts",
        )
    speech_span = active[active_indices[0] : active_indices[-1] + 1]
    internal_silence = 1.0 - float(np.mean(speech_span))

    log_energy = np.log(rms + 1e-8)
    energy_jerk = np.abs(np.diff(log_energy, n=2))
    spectra = np.abs(np.fft.rfft(windowed, axis=1))
    spectra /= np.maximum(np.linalg.norm(spectra, axis=1, keepdims=True), 1e-12)
    spectral_flux = np.linalg.norm(np.diff(spectra, axis=0), axis=1)

    minimum_lag = max(1, sample_rate // 400)
    maximum_lag = min(frame_length - 2, sample_rate // 70)
    pitches: list[float | None] = []
    for frame, is_active in zip(windowed, active, strict=True):
        if not is_active:
            pitches.append(None)
            continue
        centered = frame - float(np.mean(frame))
        correlation = np.correlate(centered, centered, mode="full")[frame_length - 1 :]
        zero_lag = float(correlation[0])
        if zero_lag <= 1e-9:
            pitches.append(None)
            continue
        region = correlation[minimum_lag : maximum_lag + 1]
        lag = minimum_lag + int(np.argmax(region))
        confidence = float(correlation[lag] / zero_lag)
        pitches.append(sample_rate / lag if confidence >= 0.35 else None)
    pitch_jumps = pitch_pairs = 0
    previous_pitch: float | None = None
    for pitch in pitches:
        if pitch is None:
            previous_pitch = None
            continue
        if previous_pitch is not None:
            pitch_pairs += 1
            pitch_jumps += abs(float(np.log2(pitch / previous_pitch))) > 0.45
        previous_pitch = pitch

    difference = np.abs(np.diff(waveform))
    peak = max(float(np.max(np.abs(waveform))), 1e-8)
    return {
        "duration_seconds": round(waveform.size / sample_rate, 4),
        "sample_rate": int(sample_rate),
        "active_speech_fraction": round(float(np.mean(active)), 6),
        "internal_silence_fraction": round(internal_silence, 6),
        "energy_jerk_median": round(float(np.median(energy_jerk)), 6),
        "energy_jerk_p95": round(float(np.percentile(energy_jerk, 95)), 6),
        "spectral_flux_median": round(float(np.median(spectral_flux)), 6),
        "spectral_flux_p95": round(float(np.percentile(spectral_flux, 95)), 6),
        "pitch_jump_rate": round(pitch_jumps / max(pitch_pairs, 1), 6),
        "voiced_pitch_pairs": pitch_pairs,
        "sample_step_p999_normalized": round(float(np.percentile(difference, 99.9)) / peak, 6),
    }


def compare_acoustic_sets(
    files: dict[str, dict[str, str | Path]],
) -> dict[str, object]:
    allowed_candidates = {
        "blend-25pct",
        "blend-37_5pct",
        "lr-2_5e-7",
        "target-freeze-output-layers",
    }
    candidates = set(files) - {"native", "baseline"}
    if (
        not candidates
        or not candidates <= allowed_candidates
        or not {"native", "baseline"} <= set(files)
        or not files["native"]
        or any(set(group) != set(files["native"]) for group in files.values())
    ):
        raise BackendError(
            "tts.greektts15.acoustic_set_incomplete",
            "GreekTTS acoustic comparison set is incomplete",
            subsystem="tts",
        )
    groups = {
        label: [
            {"fixture_id": fixture_id, **analyze_acoustic_file(path)}
            for fixture_id, path in fixtures.items()
        ]
        for label, fixtures in files.items()
    }
    aggregate: dict[str, dict[str, float]] = {}
    metrics = (
        "internal_silence_fraction",
        "energy_jerk_p95",
        "spectral_flux_p95",
        "pitch_jump_rate",
        "sample_step_p999_normalized",
    )
    for label, samples in groups.items():
        aggregate[label] = {
            metric: round(sum(float(sample[metric]) for sample in samples) / len(samples), 6)
            for metric in metrics
        }
    return {
        "schema": "jarvis.voice.greektts15-acoustic-comparison/1",
        "ready": True,
        "groups": groups,
        "aggregate_means": aggregate,
        "diagnostic_only": True,
        "human_listening_required": True,
        "promotion_allowed": False,
    }
