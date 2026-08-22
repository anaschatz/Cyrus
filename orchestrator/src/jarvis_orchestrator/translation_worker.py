"""Persistent offline CTranslate2 Greek-to-English worker."""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path
from typing import Any

from .domain import canonical_json
from .translation import (
    TRANSLATION_MAXIMUM_COMPONENTS,
    TRANSLATION_MAXIMUM_REQUEST_BYTES,
    TRANSLATION_MAXIMUM_RESPONSE_BYTES,
    TRANSLATION_MAXIMUM_TEXT_BYTES,
    TRANSLATION_REQUEST_VERSION,
    TRANSLATION_READY_VERSION,
    TRANSLATION_RESPONSE_VERSION,
    TRANSLATION_CTRANSLATE2_VERSION,
    TRANSLATION_SENTENCEPIECE_VERSION,
    _closed_json_object,
    load_translation_manifest,
)


def _valid_text(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value.strip()
        or unicodedata.normalize("NFC", value) != value
    ):
        return False
    if any(unicodedata.category(char).startswith("C") for char in value):
        return False
    try:
        return len(value.encode("utf-8", errors="strict")) <= TRANSLATION_MAXIMUM_TEXT_BYTES
    except UnicodeEncodeError:
        return False


class TranslationEngine:
    def __init__(self, model_root: Path) -> None:
        try:
            import ctranslate2
            import sentencepiece
        except ImportError as exc:
            raise RuntimeError("translation runtime unavailable") from exc
        if (
            getattr(ctranslate2, "__version__", None)
            != TRANSLATION_CTRANSLATE2_VERSION
            or getattr(sentencepiece, "__version__", None)
            != TRANSLATION_SENTENCEPIECE_VERSION
        ):
            raise RuntimeError("translation runtime version mismatch")
        self.source = sentencepiece.SentencePieceProcessor(
            model_file=str(model_root / "source.spm")
        )
        self.target = sentencepiece.SentencePieceProcessor(
            model_file=str(model_root / "target.spm")
        )
        self.translator = ctranslate2.Translator(
            str(model_root),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=1,
            max_queued_batches=1,
        )

    def translate(self, document: Any) -> dict[str, object]:
        if (
            not isinstance(document, dict)
            or set(document) != {"version", "request_id", "items"}
            or document.get("version") != TRANSLATION_REQUEST_VERSION
            or not isinstance(document.get("request_id"), str)
            or not isinstance(document.get("items"), list)
            or not 1 <= len(document["items"]) <= TRANSLATION_MAXIMUM_COMPONENTS
        ):
            raise ValueError("invalid request")
        identifiers: list[str] = []
        texts: list[str] = []
        for item in document["items"]:
            if not isinstance(item, dict) or set(item) != {"id", "text"}:
                raise ValueError("invalid request")
            identifier = item.get("id")
            text = item.get("text")
            if (
                not isinstance(identifier, str)
                or identifier in identifiers
                or not _valid_text(text)
            ):
                raise ValueError("invalid request")
            identifiers.append(identifier)
            texts.append(text)
        tokenized = [self.source.encode(text, out_type=str) for text in texts]
        for tokens in tokenized:
            if not tokens:
                raise ValueError("tokenizer returned an empty input")
            if tokens[-1] != "</s>":
                tokens.append("</s>")
        if any(len(tokens) > 512 or "<unk>" in tokens for tokens in tokenized):
            raise ValueError("input token bound exceeded")
        results = self.translator.translate_batch(
            tokenized,
            beam_size=4,
            num_hypotheses=1,
            max_input_length=512,
            max_decoding_length=512,
            batch_type="tokens",
            max_batch_size=512,
            disable_unk=True,
            return_end_token=True,
            return_scores=False,
        )
        if len(results) != len(identifiers):
            raise RuntimeError("translation result count mismatch")
        translated: list[dict[str, str]] = []
        for identifier, result in zip(identifiers, results, strict=True):
            hypotheses = getattr(result, "hypotheses", None)
            if not isinstance(hypotheses, list) or len(hypotheses) != 1:
                raise RuntimeError("translation hypothesis mismatch")
            tokens = hypotheses[0]
            if "<unk>" in tokens:
                raise RuntimeError("translation produced an unknown token")
            if not tokens or tokens[-1] != "</s>":
                raise RuntimeError("translation did not terminate")
            text = unicodedata.normalize(
                "NFC", self.target.decode_pieces(tokens[:-1]).strip()
            )
            if not _valid_text(text):
                raise RuntimeError("translation output invalid")
            translated.append({"id": identifier, "text": text})
        return {
            "version": TRANSLATION_RESPONSE_VERSION,
            "request_id": document["request_id"],
            "items": translated,
        }


def translate_document(document: Any, model_root: Path) -> dict[str, object]:
    """Compatibility helper for isolated qualification probes."""

    return TranslationEngine(model_root).translate(document)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-root", required=True)
    parser.add_argument("--manifest-path", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parsed = parser.parse_args(argv)
    model_root = Path(parsed.model_root)
    try:
        load_translation_manifest(
            model_root,
            Path(parsed.manifest_path),
            parsed.manifest_sha256,
        )
        engine = TranslationEngine(model_root)
        ready = {
            "version": TRANSLATION_READY_VERSION,
            "status": "ready",
        }
        sys.stdout.buffer.write((canonical_json(ready) + "\n").encode("utf-8"))
        sys.stdout.buffer.flush()
        while True:
            raw = sys.stdin.buffer.readline(TRANSLATION_MAXIMUM_REQUEST_BYTES + 1)
            if not raw:
                return 0
            if len(raw) > TRANSLATION_MAXIMUM_REQUEST_BYTES or not raw.endswith(b"\n"):
                return 2
            document = json.loads(
                raw.decode("utf-8"),
                object_pairs_hook=_closed_json_object,
            )
            response = engine.translate(document)
            encoded = (canonical_json(response) + "\n").encode("utf-8")
            if len(encoded) > TRANSLATION_MAXIMUM_RESPONSE_BYTES:
                return 2
            sys.stdout.buffer.write(encoded)
            sys.stdout.buffer.flush()
    except BaseException:
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
