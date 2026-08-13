from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

import _bootstrap  # noqa: F401

from jarvis_voice.errors import BackendError
from jarvis_voice.macos.greektts15_training import (
    ADAPTER_BLEND_FACTORS,
    DATASET_SCHEMA,
    EXPECTED_ROWS,
    CACHE_SCHEMA,
    build_training_plan,
    build_pronunciation_curriculum_order,
    build_lexical_curriculum_order,
    build_exact_core_curriculum_order,
    build_word_focus_bundle,
    build_word_focus_curriculum_order,
    build_word_focus_training_plan,
    blend_adapter_checkpoint,
    build_orpheus_training_sequence,
    cache_snac_samples,
    interleave_snac_codes,
    load_cached_split,
    normalize_transcript,
    score_candidate_transcripts,
    score_pronunciation_transcripts,
    score_quality_candidate_transcripts,
    _validated_gradient_probe,
    TRAINING_TARGET_PROFILES,
    TRAINING_SAMPLE_PROFILES,
    TARGETED_LORA_GROUPS,
    ATTENTION_ONLY_LORA_GROUPS,
    MLP_ONLY_LORA_GROUPS,
    EOS_LOSS_WEIGHTS,
)
from jarvis_voice.macos.greektts15_pilot import (
    END_OF_HUMAN,
    END_OF_SPEECH,
    END_OF_TEXT,
    START_OF_HUMAN,
    START_OF_SPEECH,
    extract_snac_codes,
)


class GreekTts15TrainingTests(unittest.TestCase):
    def test_word_focus_bundle_crops_only_hash_verified_real_audio(self) -> None:
        from jarvis_voice.macos.greektts15_training import WORD_FOCUS_CROPS

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / "dataset"
            audio = dataset / "audio"
            audio.mkdir(parents=True)
            samples = []
            for _, filename, _, _, word_end, _ in WORD_FOCUS_CROPS:
                path = audio / filename
                with wave.open(str(path), "wb") as stream:
                    stream.setnchannels(1)
                    stream.setsampwidth(2)
                    stream.setframerate(44_100)
                    stream.writeframes(b"\x00\x00" * round((word_end + 0.5) * 44_100))
                samples.append(
                    {
                        "filename": filename,
                        "text": "source",
                        "split": "train",
                        "audio_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            plan = root / "plan.json"
            plan.write_text(
                json.dumps({"schema": "jarvis.voice.greektts15-f1-plan/1", "samples": samples}),
                encoding="utf-8",
            )
            output = root / "focus"
            report = build_word_focus_bundle(plan.resolve(), dataset.resolve(), output.resolve())
            manifest = json.loads((output / "word-focus-manifest.json").read_text(encoding="utf-8"))
        self.assertTrue(report["ready"])
        self.assertEqual(len(manifest["samples"]), len(WORD_FOCUS_CROPS))
        self.assertTrue(all(sample["corpus"] == "STOMA-F1-word-focus" for sample in manifest["samples"]))
        self.assertTrue(all(sample["alignment_confidence"] >= 0.8 for sample in manifest["samples"]))
        self.assertTrue(all(sample["crop_start_seconds"] < sample["word_start_seconds"] for sample in manifest["samples"]))

    def test_word_focus_training_plan_preserves_validation_and_curriculum(self) -> None:
        base_samples = [
            {"split": "validation", "filename": "validation.wav", "text": "επικύρωση"},
            *(
                {"split": "train", "filename": f"general-{index}.wav", "text": f"δείγμα {index}", "corpus": "F1"}
                for index in range(30)
            ),
        ]
        focus_samples = [
            {
                "split": "train",
                "filename": f"focus-{index}.wav",
                "text": "λέξη",
                "corpus": "STOMA-F1-word-focus",
            }
            for index in range(5)
        ]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            base = root / "base.json"
            focus = root / "focus.json"
            combined = root / "combined.json"
            base.write_text(json.dumps({"schema": "jarvis.voice.greektts15-f1-plan/1", "samples": base_samples}))
            focus.write_text(
                json.dumps({"schema": "jarvis.voice.greektts15-word-focus/1", "ready": True, "samples": focus_samples})
            )
            build_word_focus_training_plan(base.resolve(), focus.resolve(), combined.resolve())
            order, report = build_word_focus_curriculum_order(combined.resolve())
            value = json.loads(combined.read_text())
        self.assertEqual(sum(sample["split"] == "validation" for sample in value["samples"]), 1)
        self.assertEqual(len(order), 25)
        self.assertEqual(report["targeted_steps"], 15)
        self.assertEqual(report["general_steps"], 10)
        self.assertEqual(len(set(order[:15])), 5)

    def test_word_focus_continuation_is_mlp_only(self) -> None:
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('sample_profile == "jarvis-word-focus" and target_profile != "mlp-only"', source)
        self.assertIn('source_adapter_directory is not None and sample_profile != "jarvis-word-focus"', source)
        self.assertIn('"source_adapter_kind": "immutable_baseline"', source)

    def test_pronunciation_curriculum_balances_real_f1_samples(self) -> None:
        samples = []
        roots = ("κατάσταση", "λειτουργούν", "θυμάμαι", "σημαντικές", "εργασίες")
        for index in range(30):
            text = f"Γενικό δείγμα {index}"
            if index < 20:
                text = f"{roots[index % len(roots)]} δείγμα {index}"
            samples.append({"split": "train", "filename": f"f{index}.wav", "text": text})
        plan = {"schema": "jarvis.voice.greektts15-f1-plan/1", "samples": samples}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            first, report = build_pronunciation_curriculum_order(path)
            second, _ = build_pronunciation_curriculum_order(path)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 25)
        self.assertEqual(len(set(first)), 25)
        self.assertEqual(report["targeted_steps"], 15)
        self.assertEqual(report["general_steps"], 10)
        self.assertTrue(all(count == 3 for count in report["target_counts"].values()))

    def test_pronunciation_sample_profile_is_attention_only(self) -> None:
        self.assertEqual(
            TRAINING_SAMPLE_PROFILES,
            {"uniform", "jarvis-pronunciation", "jarvis-lexical", "jarvis-exact-core", "jarvis-word-focus"},
        )
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('sample_profile == "jarvis-pronunciation" and target_profile != "attention-only"', source)
        self.assertIn('sample_profile == "jarvis-lexical" and target_profile != "mlp-only"', source)
        self.assertIn('sample_profile == "jarvis-exact-core" and target_profile != "mlp-only"', source)

    def test_lexical_curriculum_repeats_exact_real_examples_only(self) -> None:
        samples = []
        targets = ("κατάσταση", "λειτουργούν", "θυμάμαι", "σημαντικές", "εργασίες")
        for index in range(30):
            text = f"Γενικό δείγμα {index}"
            if index < len(targets):
                text = f"{targets[index]} δείγμα {index}"
            samples.append({"split": "train", "filename": f"f{index}.wav", "text": text})
        plan = {"schema": "jarvis.voice.greektts15-f1-plan/1", "samples": samples}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            order, report = build_lexical_curriculum_order(path)
        self.assertEqual(len(order), 25)
        self.assertEqual(report["targeted_steps"], 15)
        self.assertEqual(report["general_steps"], 10)
        self.assertTrue(report["repeated_exact_examples"])
        self.assertTrue(all(count == 3 for count in report["target_counts"].values()))
        self.assertEqual(order[:5], order[5:10])

    def test_exact_core_curriculum_excludes_reminder_proxy(self) -> None:
        targets = ("κατάσταση", "λειτουργούν", "σημαντικές", "εργασίες")
        samples = [
            {"split": "train", "filename": f"target-{index}.wav", "text": target}
            for index, target in enumerate(targets)
        ]
        samples.extend(
            {"split": "train", "filename": f"general-{index}.wav", "text": f"Γενικό δείγμα {index}"}
            for index in range(100)
        )
        plan = {"schema": "jarvis.voice.greektts15-f1-plan/1", "samples": samples}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "plan.json"
            path.write_text(json.dumps(plan), encoding="utf-8")
            order, report = build_exact_core_curriculum_order(path)
        self.assertEqual(len(order), 100)
        self.assertEqual(report["targeted_steps"], 60)
        self.assertEqual(report["general_steps"], 40)
        self.assertTrue(report["reminder_proxy_excluded"])
        self.assertEqual(set(report["target_counts"]), {"system_state", "operation", "importance", "work"})
        self.assertTrue(all(count == 15 for count in report["target_counts"].values()))

    def test_listening_candidate_manifest_cannot_claim_production_readiness(self) -> None:
        manifest = json.loads(Path("docs/greektts15-attention-only-candidate.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["schema"], "jarvis.voice.greektts15-listening-candidate/1")
        self.assertEqual(manifest["status"], "human_listening_required")
        self.assertFalse(manifest["production_promotion_allowed"])
        self.assertFalse(manifest["qualification"]["seed_robustness_passed"])
        self.assertFalse(manifest["qualification"]["human_robotic_artifact_approved"])
        self.assertEqual(len(manifest["model"]["checkpoint_sha256"]), 64)

    def test_eos_weighting_is_bounded_to_attention_only_training(self) -> None:
        self.assertEqual(EOS_LOSS_WEIGHTS, {1, 16})
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('targets == END_OF_SPEECH', source)
        self.assertIn('eos_loss_weight != 1 and target_profile != "attention-only"', source)

    def test_quality_candidate_scoring_stays_baseline_relative(self) -> None:
        references = {"one": "Καλημέρα κόσμε"}
        report = score_quality_candidate_transcripts(
            references,
            {
                "baseline": {"one": "Καλημέρα"},
                "attention-only": dict(references),
            },
        )
        self.assertTrue(report["candidate_not_worse_than_baseline"])
        self.assertFalse(report["promotion_allowed"])

    def test_targeted_profile_freezes_exactly_two_complete_adapter_layers(self) -> None:
        self.assertEqual(TRAINING_TARGET_PROFILES, {"all", "freeze-output-layers", "attention-only", "mlp-only"})
        self.assertEqual(TARGETED_LORA_GROUPS, 26 * 7 * 2)
        self.assertEqual(ATTENTION_ONLY_LORA_GROUPS, 28 * 4 * 2)
        self.assertEqual(MLP_ONLY_LORA_GROUPS, 28 * 3 * 2)
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('frozen_layers = [26, 27]', source)
        self.assertIn('len(complete_adapter) != EXPECTED_LORA_GROUPS', source)
        self.assertIn('layer.mlp.freeze()', source)
        self.assertIn('layer.self_attn.freeze()', source)

    def test_pronunciation_asr_calibrates_against_native_f1(self) -> None:
        references = {"one": "Καλημέρα κόσμε", "two": "Όλα λειτουργούν"}
        transcripts = {
            "native": dict(references),
            "baseline": {"one": "Καλημέρα", "two": "Όλα λειτουργούν"},
            "blend-25pct": dict(references),
        }
        report = score_pronunciation_transcripts(references, transcripts)
        self.assertEqual(report["native_calibration_wer_percent"], 0.0)
        self.assertEqual(report["candidate_label"], "blend-25pct")
        self.assertTrue(report["candidate_not_worse_than_baseline"])
        self.assertFalse(report["promotion_allowed"])

    def test_pronunciation_asr_accepts_only_bounded_blend_labels(self) -> None:
        references = {"one": "Καλημέρα"}
        transcripts = {
            "native": dict(references),
            "baseline": dict(references),
            "blend-37_5pct": dict(references),
        }
        self.assertEqual(
            score_pronunciation_transcripts(references, transcripts)["candidate_label"],
            "blend-37_5pct",
        )
        transcripts["unknown"] = transcripts.pop("blend-37_5pct")
        with self.assertRaises(BackendError):
            score_pronunciation_transcripts(references, transcripts)

        transcripts["lr-2_5e-7"] = transcripts.pop("unknown")
        self.assertEqual(
            score_pronunciation_transcripts(references, transcripts)["candidate_label"],
            "lr-2_5e-7",
        )
        transcripts["target-freeze-output-layers"] = transcripts.pop("lr-2_5e-7")
        self.assertEqual(
            score_pronunciation_transcripts(references, transcripts)["candidate_label"],
            "target-freeze-output-layers",
        )
        transcripts["attention-only"] = transcripts.pop("target-freeze-output-layers")
        self.assertEqual(
            score_pronunciation_transcripts(references, transcripts)["candidate_label"],
            "attention-only",
        )

    def test_adapter_blending_is_bounded_away_from_rejected_candidate(self) -> None:
        self.assertEqual(ADAPTER_BLEND_FACTORS, {0.25, 0.375, 0.5})
        with self.assertRaises(BackendError) as raised:
            blend_adapter_checkpoint("/models", "/candidate", "/output", candidate_fraction=0.75)
        self.assertEqual(raised.exception.code, "tts.greektts15.blend_factor_invalid")

    def test_adapter_blending_requires_new_absolute_output(self) -> None:
        with self.assertRaises(BackendError) as raised:
            blend_adapter_checkpoint("relative", "/candidate", "/output", candidate_fraction=0.25)
        self.assertEqual(raised.exception.code, "tts.greektts15.blend_path_invalid")

    def test_candidate_asr_scoring_compares_identical_fixture_sets(self) -> None:
        references = {"one": "Καλημέρα κόσμε", "two": "Όλα λειτουργούν"}
        transcripts = {
            "baseline": dict(references),
            "lr-5e-7": {"one": "Καλημέρα", "two": "Όλα λειτουργούν"},
            "lr-1e-6": dict(references),
        }
        report = score_candidate_transcripts(references, transcripts)
        by_label = {item["label"]: item for item in report["candidates"]}
        self.assertEqual(by_label["baseline"]["micro_wer_percent"], 0.0)
        self.assertGreater(by_label["lr-5e-7"]["micro_wer_percent"], 0.0)
        self.assertTrue(report["human_listening_required"])
        self.assertFalse(report["promotion_allowed"])

    def test_optimizer_gate_accepts_only_no_update_memory_probe(self) -> None:
        valid = {
            "schema": "jarvis.voice.greektts15-gradient-probe/1",
            "ready": True,
            "optimizer_update_applied": False,
            "trainable_groups": 392,
            "gradient_groups": 392,
            "gradient_checkpointing": True,
            "sequence_selection": "longest_train_sample",
            "peak_mlx_bytes": 4_000_000_000,
        }
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp) / "probe.json"
            report.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(_validated_gradient_probe(report), valid)
            valid["optimizer_update_applied"] = True
            report.write_text(json.dumps(valid), encoding="utf-8")
            with self.assertRaises(BackendError) as raised:
                _validated_gradient_probe(report)
        self.assertEqual(raised.exception.code, "tts.greektts15.gradient_gate_failed")

    def test_memory_probe_source_selects_longest_training_sequence(self) -> None:
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('max(dataset, key=lambda item: len(item[0]))', source)
        self.assertIn('grad_checkpoint(model.layers[0])', source)

    def test_pilot_is_bounded_to_two_learning_rates_and_25_steps(self) -> None:
        source = Path("src/jarvis_voice/macos/greektts15_training.py").read_text(encoding="utf-8")
        self.assertIn('learning_rate not in PILOT_LEARNING_RATES', source)
        self.assertIn('PILOT_LEARNING_RATES = {2.5e-7, 5e-7, 1e-6, 5e-6, 1e-5}', source)
        self.assertIn('steps == EXACT_CORE_STEPS and sample_profile == "jarvis-exact-core"', source)
        self.assertIn('random.Random(TRAINING_SEED)', source)

    def test_cached_split_builds_completion_offset(self) -> None:
        class Tokenizer:
            def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
                self.assertion = (text, add_special_tokens)
                return [128000, 42]

        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp) / "dataset")
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                plan = build_training_plan(root)
            plan_path = Path(temp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            cache = Path(temp) / "cache"
            encoder = self._FakeEncoder()
            cache_snac_samples(plan_path, root, cache, encoder)
            dataset = load_cached_split(plan_path, cache, Tokenizer(), "train")
        sequence, offset = dataset[0]
        self.assertEqual(sequence[offset], START_OF_SPEECH)
        self.assertEqual(offset, 5)

    def test_snac_cache_is_resumable_and_content_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp) / "dataset")
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                plan = build_training_plan(root)
            plan_path = Path(temp) / "plan.json"
            plan_path.write_text(json.dumps(plan), encoding="utf-8")
            encoder = self._FakeEncoder()
            cache_root = Path(temp) / "cache"
            first = cache_snac_samples(plan_path, root, cache_root, encoder, limit=2)
            second = cache_snac_samples(plan_path, root, cache_root, encoder, limit=2)
            payload = json.loads(next(cache_root.iterdir()).read_text())
        self.assertEqual(first["schema"], CACHE_SCHEMA)
        self.assertEqual((first["encoded"], first["reused"]), (2, 0))
        self.assertEqual((second["encoded"], second["reused"]), (0, 2))
        self.assertEqual(encoder.calls, 2)
        self.assertEqual(payload["encoder"], encoder.signature)

    class _FakeEncoder:
        signature = {"backend": "fake", "revision": "test"}

        def __init__(self) -> None:
            self.calls = 0

        def encode(self, audio_path: Path) -> list[int]:
            self.calls += 1
            del audio_path
            return interleave_snac_codes([1], [2, 3], [4, 5, 6, 7])

    def test_snac_interleave_round_trips_through_runtime_decoder(self) -> None:
        layers = ([1, 2], [3, 4, 5, 6], list(range(7, 15)))
        audio = interleave_snac_codes(*layers)
        self.assertEqual(extract_snac_codes([START_OF_SPEECH, *audio, END_OF_SPEECH]), layers)

    def test_training_sequence_matches_runtime_control_protocol(self) -> None:
        audio = interleave_snac_codes([1], [2, 3], [4, 5, 6, 7])
        ids, labels = build_orpheus_training_sequence([128000, 42], audio)
        prefix = [START_OF_HUMAN, 128000, 42, END_OF_TEXT, END_OF_HUMAN]
        completion = [START_OF_SPEECH, *audio, END_OF_SPEECH]
        self.assertEqual(ids, prefix + completion)
        self.assertEqual(labels, [-100] * len(prefix) + completion)

    def test_training_sequence_does_not_duplicate_eot(self) -> None:
        audio = interleave_snac_codes([1], [2, 3], [4, 5, 6, 7])
        ids, _ = build_orpheus_training_sequence([128000, END_OF_TEXT], audio)
        self.assertEqual(ids.count(END_OF_TEXT), 1)

    def test_normalization_repairs_only_leading_greek_confusables(self) -> None:
        self.assertEqual(normalize_transcript("Tην  ίδια μέρα"), ("Την ίδια μέρα", True))
        self.assertEqual(normalize_transcript("To παιδί"), ("Το παιδί", True))
        self.assertEqual(normalize_transcript("Η CPU λειτουργεί"), ("Η CPU λειτουργεί", False))

    def test_plan_has_deterministic_stratified_splits(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp))
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                first = build_training_plan(root)
                second = build_training_plan(root)
        self.assertEqual(first, second)
        self.assertEqual(first["split"]["counts"], {"train": 16, "validation": 2, "control": 2})
        self.assertFalse(first["training_policy"]["merge_into_quantized_base"])
        self.assertTrue(first["training_policy"]["owned_greektts_token_collator"])

    def test_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = self._bundle(Path(temp))
            target = next((root / "audio").iterdir())
            target.write_bytes(b"changed")
            with mock.patch("jarvis_voice.macos.greektts15_training.EXPECTED_ROWS", 20):
                with self.assertRaises(BackendError) as raised:
                    build_training_plan(root)
        self.assertEqual(raised.exception.code, "tts.greektts15.dataset_audio_hash_invalid")

    @staticmethod
    def _bundle(root: Path) -> Path:
        audio = root / "audio"
        audio.mkdir(parents=True)
        rows = []
        hashes = {}
        for corpus in ("B2", "C1"):
            for index in range(10):
                filename = f"Greek-F1-{corpus}-01-{index:02d}.wav"
                path = audio / filename
                with wave.open(str(path), "wb") as wav:
                    wav.setnchannels(1)
                    wav.setsampwidth(2)
                    wav.setframerate(44_100)
                    wav.writeframes(b"\0\0" * (44_100 * 2))
                hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                rows.append(f"{filename}|Tο δείγμα {index}.")
        metadata = root / "metadata.csv"
        metadata.write_text("\n".join(rows) + "\n", encoding="utf-8")
        (root / "bundle-manifest.json").write_text(
            json.dumps(
                {
                    "schema": DATASET_SCHEMA,
                    "rows": len(rows),
                    "dataset_revision": "test",
                    "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
                    "audio_sha256": hashes,
                }
            ),
            encoding="utf-8",
        )
        return root


if __name__ == "__main__":
    unittest.main()
