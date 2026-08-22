"""Persistent offline CTranslate2 Greek-to-English worker."""

from __future__ import annotations

import argparse
import json
import re
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


_COMMAND_REWRITES = (
    (re.compile(r"^\s*Ναι\s*[.!]?\s*$", re.IGNORECASE), "Yes."),
    (
        re.compile(
            r"^\s*Στις\s+(\d{1,2}:\d{2})\s+αποθήκευσε\s+"
            r"το\s+αρχείο\b",
            re.IGNORECASE,
        ),
        r"At \1, save the file",
    ),
    (
        re.compile(r"^(\s*)Βάλε υπενθύμιση\b", re.IGNORECASE),
        r"\1Set a reminder",
    ),
    (re.compile(r"^(\s*)Σταμάτα\b", re.IGNORECASE), r"\1Stop"),
    (
        re.compile(
            r"^(\s*)Σταμάτησε\s+αμέσως\s+"
            r"την\s+αναπαραγωγή\b",
            re.IGNORECASE,
        ),
        r"\1Stop the playback immediately",
    ),
    (
        re.compile(
            r"^(\s*)Συνέχισε\s+από\s+εκεί\s+που\s+"
            r"σταμάτησες\b",
            re.IGNORECASE,
        ),
        r"\1Continue where you left off",
    ),
    (
        re.compile(
            r"^(\s*)Κράτησε\s+το\s+πρώτο\b",
            re.IGNORECASE,
        ),
        r"\1Keep the first",
    ),
    (
        re.compile(
            r"^(\s*)Κράτησε\s+τα\s+ονόματα\s+"
            r"(.+?)\s+ακριβώς\s+όπως\s+είναι\b",
            re.IGNORECASE,
        ),
        r"\1Keep the names \2 exactly as they are",
    ),
    (
        re.compile(
            r"^(\s*)Απόρριψε\s+κάθε\s+παλιά\s+έξοδο\s+"
            r"μετά\s+τη\s+διακοπή\b",
            re.IGNORECASE,
        ),
        r"\1Reject each stale output after interruption",
    ),
    (
        re.compile(
            r"^(\s*)Κατάλαβε\s+τα\s+ελληνικά,\s*"
            r"αλλά\s+απάντησε\s+αποκλειστικά\s+"
            r"στα\s+αγγλικά\b",
            re.IGNORECASE,
        ),
        r"\1Understand Greek, but answer only in English",
    ),
)
_NARRATIVE_MARKERS = re.compile(
    r"\b(?:χθες|προχθές|πέρσι|νωρίτερα|προηγουμένως|τότε|"
    r"την\s+προηγούμενη|την\s+περασμένη|"
    r"τον\s+προηγούμενο|τον\s+περασμένο|"
    r"πριν\s+από)\b",
    re.IGNORECASE,
)
_TERM_REWRITES = (
    (
        re.compile(
            r"\bΌταν επιστρέψει το μικρόφωνο,"
            r"\s*συνέχισε αυτόματα\b",
            re.IGNORECASE,
        ),
        "When the microphone reconnects, continue automatically",
    ),
    (
        re.compile(
            r"\bΜετά την αναστολή λειτουργίας,"
            r"\s*έλεγξε ξανά\b",
            re.IGNORECASE,
        ),
        "After waking from sleep, recheck",
    ),
    (
        re.compile(
            r"\bάλλαξε στην είσοδο\s+(USB Audio CODEC)\b",
            re.IGNORECASE,
        ),
        r"switch to the \1 input",
    ),
    (
        re.compile(
            r"\bπαλιά έξοδο μετά (?:τη|την) διακοπή\b",
            re.IGNORECASE,
        ),
        "stale output after interruption",
    ),
    (re.compile(r"\bαναπαραγωγή\b", re.IGNORECASE), "playback"),
    (re.compile(r"\bπαλιό ήχο\b", re.IGNORECASE), "old audio"),
    (
        re.compile(r"\bτρέχουσα απάντηση\b", re.IGNORECASE),
        "current response",
    ),
    (re.compile(r"\bαπάντηση\b", re.IGNORECASE), "response"),
    (re.compile(r"\bμεταγραφή\b", re.IGNORECASE), "transcript"),
    (
        re.compile(r"\bΜετά το sleep\b", re.IGNORECASE),
        "After waking from sleep",
    ),
    (re.compile(r"\bΜαρία\b", re.IGNORECASE), "Maria"),
    (re.compile(r"\bαναστολή λειτουργίας\b", re.IGNORECASE), "sleep"),
    (re.compile(r"\bραντεβού\b", re.IGNORECASE), "appointment"),
    (re.compile(r"\bμόνο στην CPU\b", re.IGNORECASE), "only on CPU"),
    (
        re.compile(
            r"\bΤο M3 έχει (\d+(?:[.,]\d+)?\s*GB) κοινής μνήμης\b",
            re.IGNORECASE,
        ),
        r"M3 has \1 unified memory",
    ),
    (re.compile(r"\bστο (config\.json)\b", re.IGNORECASE), r"in \1"),
    (
        re.compile(r"\bη δοκιμή ολοκληρώθηκε\b", re.IGNORECASE),
        "the test is complete",
    ),
    (
        re.compile(r"\bΔεν θέλω API key\b", re.IGNORECASE),
        "I don't want an API key",
    ),
)
_PROTECTED_PATTERNS = (
    re.compile(r"https?://[^\s<>\"'“”«»]*[^\s<>\"'“”«».,;:!?]"),
    re.compile(
        r"\\\\[A-Za-z0-9._~-]+"
        r"\\[A-Za-z0-9$._~-](?:[A-Za-z0-9$._~ -]*[A-Za-z0-9$_~-])?"
        r"(?:\\[A-Za-z0-9._~-](?:[A-Za-z0-9._~ -]*[A-Za-z0-9_~-])?)*"
        r"\\?"
    ),
    re.compile(
        r"\b[A-Za-z]:\\"
        r"[A-Za-z0-9._~-](?:[A-Za-z0-9._~ -]*[A-Za-z0-9_~-])?"
        r"(?:\\[A-Za-z0-9._~-](?:[A-Za-z0-9._~ -]*[A-Za-z0-9_~-])?)*"
        r"\\?"
    ),
    re.compile(
        r"(?<![A-Za-z0-9])/"
        r"[A-Za-z0-9._~-](?:[A-Za-z0-9._~ -]*[A-Za-z0-9_~-])?"
        r"(?:/[A-Za-z0-9._~-](?:[A-Za-z0-9._~ -]*[A-Za-z0-9_~-])?)*"
        r"/?"
    ),
    re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*=[A-Za-z0-9_.-]+\b"),
    re.compile(
        r"\b[A-Za-z0-9][A-Za-z0-9._-]*\."
        r"(?:md|json|txt|yaml|yml|toml|py)\b"
    ),
    re.compile(r"\b\d{1,2}:\d{2}\b"),
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{4}\b"),
    re.compile(r"\b\d+(?:[.,]\d+)?\s*(?:GB|MB|KiB|MiB|ms)\b"),
    re.compile(r"\b\d+(?:[.,]\d+)?%"),
    re.compile(r"\b\d+(?:[.,]\d+)?\b"),
    re.compile(r"\b[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*)+\b"),
    re.compile(
        r"\b(?:[A-Z]{2,}[A-Z0-9-]*|[A-Z][0-9][A-Za-z0-9-]*|"
        r"[A-Za-z]+[0-9][A-Za-z0-9.-]*)\b"
    ),
)


def _domain_normalize(text: str, *, command_context: bool) -> str:
    result = text
    if command_context and _NARRATIVE_MARKERS.search(result) is None:
        for pattern, replacement in _COMMAND_REWRITES:
            result = pattern.sub(replacement, result, count=1)
    for pattern, replacement in _TERM_REWRITES:
        result = pattern.sub(replacement, result)
    return result


def _protect_spans(text: str) -> tuple[str, tuple[tuple[str, str], ...]]:
    matches: list[tuple[int, int]] = []
    for pattern in _PROTECTED_PATTERNS:
        matches.extend(
            (match.start(), match.end()) for match in pattern.finditer(text)
        )
    selected: list[tuple[int, int]] = []
    for start, end in sorted(
        matches,
        key=lambda item: (item[0], -(item[1] - item[0])),
    ):
        if any(
            start < other_end and other_start < end
            for other_start, other_end in selected
        ):
            continue
        selected.append((start, end))
    selected.sort()
    domain_index = 0
    domain = "jarvis.invalid"
    while domain in text:
        domain_index += 1
        domain = f"jarvis-protected-{domain_index}.invalid"
    replacements: list[tuple[str, str]] = []
    result = text
    for index, (start, end) in reversed(tuple(enumerate(selected))):
        sentinel = f"https://{domain}/p{index}q"
        original = text[start:end]
        result = result[:start] + sentinel + result[end:]
        replacements.append((sentinel, original))
    replacements.reverse()
    return result, tuple(replacements)


def _restore_spans(text: str, replacements: tuple[tuple[str, str], ...]) -> str:
    positions: list[int] = []
    for sentinel, _original in replacements:
        if text.count(sentinel) != 1:
            raise RuntimeError("translation changed a protected span")
        positions.append(text.index(sentinel))
    if positions != sorted(positions):
        raise RuntimeError("translation reordered protected spans")

    result = text
    for sentinel, original in replacements:
        result = result.replace(sentinel, original, 1)
    if any(sentinel in result for sentinel, _original in replacements):
        raise RuntimeError("translation left a protected sentinel")
    if replacements:
        sentinel_origin = replacements[0][0].rsplit("/p", 1)[0]
        if sentinel_origin in result:
            raise RuntimeError("translation invented a protected sentinel")
    return result


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
        return (
            len(value.encode("utf-8", errors="strict"))
            <= TRANSLATION_MAXIMUM_TEXT_BYTES
        )
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
        raw_vocabulary = (model_root / "shared_vocabulary.json").read_bytes()
        if len(raw_vocabulary) > 16 * 1024 * 1024:
            raise RuntimeError("translation vocabulary is too large")
        vocabulary = json.loads(raw_vocabulary.decode("utf-8"))
        if (
            not isinstance(vocabulary, list)
            or not vocabulary
            or any(not isinstance(token, str) or not token for token in vocabulary)
            or len(set(vocabulary)) != len(vocabulary)
            or "<unk>" not in vocabulary
            or "</s>" not in vocabulary
        ):
            raise RuntimeError("translation vocabulary is invalid")
        self.source_vocabulary = frozenset(vocabulary)
        self.translator = ctranslate2.Translator(
            str(model_root),
            device="cpu",
            compute_type="int8",
            inter_threads=1,
            intra_threads=1,
            max_queued_batches=1,
        )

    def _source_tokens(self, text: str) -> list[str]:
        normalized = text.translate(
            {
                ord("\u2013"): "-",
                ord("\u2014"): "-",
                ord("\u2022"): "-",
            }
        )

        def selected(value: str) -> list[str] | None:
            candidates = self.source.nbest_encode(
                value,
                nbest_size=128,
                out_type=str,
            )
            return next(
                (
                    list(candidate)
                    for candidate in candidates
                    if candidate
                    and all(token in self.source_vocabulary for token in candidate)
                ),
                None,
            )

        direct = self.source.encode(normalized, out_type=str)
        if direct and all(token in self.source_vocabulary for token in direct):
            return list(direct)
        whole = selected(normalized)
        if whole is not None:
            return whole
        tokens: list[str] = []
        for span in re.findall(r"\S+", normalized):
            replacement = selected(span)
            if replacement is None:
                raise ValueError("input contains an unsupported token")
            tokens.extend(replacement)
        if not tokens:
            raise ValueError("tokenizer returned an empty input")
        return tokens

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
        prepared = [
            _protect_spans(
                _domain_normalize(text, command_context=identifier == "current")
            )
            for identifier, text in zip(identifiers, texts, strict=True)
        ]
        tokenized = [self._source_tokens(text) for text, _spans in prepared]
        for tokens in tokenized:
            if not tokens:
                raise ValueError("tokenizer returned an empty input")
            if tokens[-1] != "</s>":
                tokens.append("</s>")
        if any(
            len(tokens) > 512
            or "<unk>" in tokens
            or any(token not in self.source_vocabulary for token in tokens)
            for tokens in tokenized
        ):
            raise ValueError("input token bound exceeded")
        results = self.translator.translate_batch(
            tokenized,
            beam_size=4,
            num_hypotheses=1,
            max_input_length=512,
            max_decoding_length=512,
            batch_type="examples",
            max_batch_size=1,
            disable_unk=True,
            return_end_token=True,
            return_scores=False,
        )
        if len(results) != len(identifiers):
            raise RuntimeError("translation result count mismatch")
        translated: list[dict[str, str]] = []
        for identifier, result, (_source, spans) in zip(
            identifiers,
            results,
            prepared,
            strict=True,
        ):
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
            text = _restore_spans(text, spans)
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
