from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import _bootstrap  # noqa: F401
import jarvis_voice.voice_auth as voice_auth

from jarvis_voice.voice_auth import (
    AUTH_PROTOCOL,
    CHALLENGE_LIFETIME_MS,
    KEY_FILE_SCHEMA,
    MAX_AUTH_FRAME_BYTES,
    MAX_HELLO_FRAME_BYTES,
    VoiceAuthError,
    VoiceAuthKey,
    create_client_response,
    create_server_challenge,
    create_server_ready,
    create_voice_auth_key_file,
    load_voice_auth_key,
    parse_client_response,
    parse_server_challenge,
    parse_server_ready,
    verify_hello_binding,
)


NOW_MS = 1_800_000_000_000
CONNECTION_ID = "123e4567-e89b-42d3-a456-426614174000"
KEY = VoiceAuthKey(7, bytes(range(32)))
OTHER_KEY = VoiceAuthKey(7, bytes(reversed(range(32))))
SERVER_NONCE = bytes(range(32, 64))
CLIENT_NONCE = bytes(range(64, 96))
HELLO = b'{"payload":{"capabilities":[],"protocol":"jarvis.voice/1"},"seq":1,"type":"client.hello"}'


def changed_json(raw: bytes, field: str, value: object) -> bytes:
    document = json.loads(raw)
    document[field] = value
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode("ascii")


class VoiceAuthHandshakeTests(unittest.TestCase):
    def test_key_repr_does_not_disclose_secret_bytes(self) -> None:
        self.assertEqual(repr(KEY), "VoiceAuthKey(key_version=7)")

    def handshake(self):
        challenge = create_server_challenge(
            KEY,
            connection_id=CONNECTION_ID,
            now_ms=NOW_MS,
            server_nonce=SERVER_NONCE,
        )
        challenge_raw = challenge.to_bytes()
        response = create_client_response(
            KEY,
            challenge_raw,
            HELLO,
            now_ms=NOW_MS + 1,
            client_nonce=CLIENT_NONCE,
        )
        response_raw = response.to_bytes()
        ready = create_server_ready(
            KEY,
            challenge_raw,
            response_raw,
            HELLO,
            now_ms=NOW_MS + 2,
        )
        return challenge, challenge_raw, response, response_raw, ready

    def test_complete_mutual_handshake_round_trip(self) -> None:
        challenge, challenge_raw, response, response_raw, ready = self.handshake()

        self.assertEqual(challenge.connection_id, CONNECTION_ID)
        self.assertEqual(challenge.expires_at_ms - challenge.issued_at_ms, CHALLENGE_LIFETIME_MS)
        self.assertEqual(parse_server_challenge(challenge_raw, KEY, now_ms=NOW_MS), challenge)
        self.assertEqual(
            parse_client_response(response_raw, KEY, challenge_raw, now_ms=NOW_MS + 1),
            response,
        )
        self.assertEqual(verify_hello_binding(response, HELLO), HELLO)
        self.assertEqual(
            parse_server_ready(
                ready.to_bytes(),
                KEY,
                challenge_raw,
                response_raw,
                HELLO,
                now_ms=NOW_MS + 2,
            ),
            ready,
        )
        self.assertEqual(json.loads(challenge_raw)["protocol"], AUTH_PROTOCOL)

    def test_wrong_key_fails_both_server_and_client_authentication(self) -> None:
        _challenge, challenge_raw, _response, response_raw, _ready = self.handshake()
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.authentication_failed"):
            parse_server_challenge(challenge_raw, OTHER_KEY, now_ms=NOW_MS)
        bad_response = create_client_response(
            OTHER_KEY,
            create_server_challenge(
                OTHER_KEY,
                connection_id=CONNECTION_ID,
                now_ms=NOW_MS,
                server_nonce=SERVER_NONCE,
            ).to_bytes(),
            HELLO,
            now_ms=NOW_MS,
            client_nonce=CLIENT_NONCE,
        )
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.authentication_failed"):
            parse_client_response(bad_response.to_bytes(), KEY, challenge_raw, now_ms=NOW_MS)
        self.assertNotEqual(response_raw, bad_response.to_bytes())

    def test_challenge_is_exactly_five_seconds_and_time_bounded(self) -> None:
        challenge = create_server_challenge(
            KEY,
            connection_id=CONNECTION_ID,
            now_ms=NOW_MS,
            server_nonce=SERVER_NONCE,
        )
        parse_server_challenge(challenge.to_bytes(), KEY, now_ms=NOW_MS + 5_000)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.challenge_expired"):
            parse_server_challenge(challenge.to_bytes(), KEY, now_ms=NOW_MS + 5_001)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.challenge_not_yet_valid"):
            parse_server_challenge(challenge.to_bytes(), KEY, now_ms=NOW_MS - 1)

        invalid_lifetime = changed_json(
            challenge.to_bytes(), "expires_at_ms", NOW_MS + 5_001
        )
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.challenge_time_invalid"):
            parse_server_challenge(invalid_lifetime, KEY, now_ms=NOW_MS)

    def test_response_binds_connection_key_and_server_nonce(self) -> None:
        _challenge, challenge_raw, _response, response_raw, _ready = self.handshake()
        variants = (
            changed_json(response_raw, "connection_id", str(uuid.uuid4())),
            changed_json(response_raw, "key_version", 8),
            changed_json(response_raw, "server_nonce", "A" * 43),
        )
        for variant in variants:
            with self.subTest(variant=variant):
                with self.assertRaises(VoiceAuthError):
                    parse_client_response(variant, KEY, challenge_raw, now_ms=NOW_MS + 1)

    def test_hello_digest_commits_to_exact_bytes(self) -> None:
        _challenge, challenge_raw, response, _response_raw, _ready = self.handshake()
        verify_hello_binding(response, HELLO)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.hello_mismatch"):
            verify_hello_binding(response, HELLO + b" ")
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.frame_too_large"):
            verify_hello_binding(response, b"x" * (MAX_HELLO_FRAME_BYTES + 1))

    def test_server_ready_binds_every_transcript_frame(self) -> None:
        _challenge, challenge_raw, _response, response_raw, ready = self.handshake()
        arguments = (ready.to_bytes(), KEY, challenge_raw, response_raw, HELLO)
        parse_server_ready(*arguments, now_ms=NOW_MS + 2)

        tampered_response = response_raw.replace(b'"client_nonce":"', b' "client_nonce":"', 1)
        # The altered whitespace remains valid JSON and preserves response MAC
        # semantics, but must change the exact transcript MAC.
        parse_client_response(tampered_response, KEY, challenge_raw, now_ms=NOW_MS + 2)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.authentication_failed"):
            parse_server_ready(
                ready.to_bytes(),
                KEY,
                challenge_raw,
                tampered_response,
                HELLO,
                now_ms=NOW_MS + 2,
            )
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.hello_mismatch"):
            parse_server_ready(
                ready.to_bytes(),
                KEY,
                challenge_raw,
                response_raw,
                HELLO + b" ",
                now_ms=NOW_MS + 2,
            )

    def test_unknown_missing_duplicate_and_oversized_fields_are_rejected(self) -> None:
        challenge, challenge_raw, _response, _response_raw, _ready = self.handshake()
        unknown = json.loads(challenge_raw)
        unknown["future"] = True
        missing = json.loads(challenge_raw)
        del missing["server_nonce"]
        duplicate = challenge_raw[:-1] + b',"key_version":7}'
        for raw in (
            json.dumps(unknown).encode(),
            json.dumps(missing).encode(),
            duplicate,
        ):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(VoiceAuthError, "voice_auth.frame_invalid"):
                    parse_server_challenge(raw, KEY, now_ms=NOW_MS)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.frame_too_large"):
            parse_server_challenge(b"{" + b" " * MAX_AUTH_FRAME_BYTES + b"}", KEY, now_ms=NOW_MS)
        self.assertEqual(challenge.key_version, 7)

    def test_deep_json_is_rejected_by_all_handshake_parsers(self) -> None:
        _, challenge, _, response, _ = self.handshake()
        malformed = b'{"nested":' + b'[' * 1200 + b'0' + b']' * 1200 + b'}'
        for parser, arguments in (
            (parse_server_challenge, (KEY,)),
            (parse_client_response, (KEY, challenge)),
            (parse_server_ready, (KEY, challenge, response, HELLO)),
        ):
            with self.subTest(parser=parser.__name__):
                with self.assertRaisesRegex(VoiceAuthError, "voice_auth.frame_invalid"):
                    parser(malformed, *arguments, now_ms=NOW_MS)

    def test_noncanonical_or_wrong_length_base64_is_rejected(self) -> None:
        _challenge, challenge_raw, _response, _response_raw, _ready = self.handshake()
        for value in ("A" * 42, "A" * 42 + "=", "+" + "A" * 42):
            with self.subTest(value=value):
                with self.assertRaisesRegex(VoiceAuthError, "voice_auth.base64_invalid"):
                    parse_server_challenge(
                        changed_json(challenge_raw, "server_nonce", value),
                        KEY,
                        now_ms=NOW_MS,
                    )

    def test_protocol_type_uuid_and_boolean_integer_are_strict(self) -> None:
        _challenge, challenge_raw, _response, _response_raw, _ready = self.handshake()
        cases = (
            ("protocol", "jarvis.voice-auth/2", "voice_auth.protocol_mismatch"),
            ("type", "client.response", "voice_auth.type_invalid"),
            ("connection_id", CONNECTION_ID.upper(), "voice_auth.binding_invalid"),
            ("key_version", True, "voice_auth.binding_invalid"),
            ("issued_at_ms", True, "voice_auth.challenge_time_invalid"),
        )
        for field, value, code in cases:
            with self.subTest(field=field):
                with self.assertRaisesRegex(VoiceAuthError, code):
                    parse_server_challenge(
                        changed_json(challenge_raw, field, value), KEY, now_ms=NOW_MS
                    )

    def test_authentication_codes_never_contain_frame_or_key_material(self) -> None:
        _challenge, challenge_raw, _response, _response_raw, _ready = self.handshake()
        tampered = changed_json(challenge_raw, "server_mac", "A" * 43)
        try:
            parse_server_challenge(tampered, KEY, now_ms=NOW_MS)
        except VoiceAuthError as exc:
            self.assertEqual(str(exc), exc.code)
            self.assertNotIn(CONNECTION_ID, str(exc))
            self.assertNotIn("AAAA", str(exc))
        else:  # pragma: no cover - assertion clarity
            self.fail("tampered MAC was accepted")


class VoiceAuthKeyFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        if os.name == "nt":  # pragma: no cover - native Windows CI
            voice_auth._harden_windows_key_file(self.root)
        self.path = self.root / "voice-auth.json"

    def write(self, raw: bytes, *, mode: int = 0o600) -> None:
        self.path.write_bytes(raw)
        if os.name == "posix":
            self.path.chmod(mode)
        elif os.name == "nt":  # pragma: no cover - native Windows CI
            voice_auth._harden_windows_key_file(self.path)

    def test_create_and_load_exact_versioned_key_file(self) -> None:
        expected = bytes(range(32))
        created = create_voice_auth_key_file(self.path, key_version=9, key=expected)
        loaded = load_voice_auth_key(self.path)
        document = json.loads(self.path.read_text(encoding="utf-8"))

        self.assertEqual(created, VoiceAuthKey(9, expected))
        self.assertEqual(loaded, created)
        self.assertEqual(set(document), {"schema_version", "key_version", "key_b64"})
        self.assertEqual(document["schema_version"], KEY_FILE_SCHEMA)
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_create_is_exclusive_and_does_not_replace_existing_file(self) -> None:
        first = create_voice_auth_key_file(self.path, key=bytes(range(32)))
        before = self.path.read_bytes()
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_create_failed"):
            create_voice_auth_key_file(self.path, key=bytes(reversed(range(32))))
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(load_voice_auth_key(self.path), first)

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO substitution")
    def test_key_reader_rejects_fifo_substitution_without_blocking(self) -> None:
        create_voice_auth_key_file(self.path, key=bytes(range(32)))
        script = """
import os, sys
from pathlib import Path
from unittest.mock import patch
from jarvis_voice.voice_auth import VoiceAuthError, load_voice_auth_key
path = Path(sys.argv[1]); real_open = os.open
def substitute(name, flags, *args, **kwargs):
    path.unlink()
    os.mkfifo(path, 0o600)
    return real_open(name, flags, *args, **kwargs)
with patch.object(os, "open", side_effect=substitute):
    try:
        load_voice_auth_key(path)
    except VoiceAuthError:
        pass
    else:
        raise AssertionError("FIFO accepted")
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(self.path)],
            capture_output=True, text=True, timeout=3,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_failed_key_creation_preserves_replacement_files(self) -> None:
        for phase in ("fsync", "harden", "interrupt"):
            for replace_file in (False, True):
                with self.subTest(phase=phase, replace_file=replace_file):
                    def fail(*args):
                        if replace_file:
                            self.path.unlink()
                            self.path.write_bytes(b"replacement")
                        if phase == "interrupt":
                            raise KeyboardInterrupt
                        if phase == "fsync":
                            raise OSError("simulated write failure")
                        raise VoiceAuthError("voice_auth.key_permissions_invalid")
                    target = "jarvis_voice.voice_auth." + (
                        "_harden_windows_key_file" if phase == "harden" else "os.fsync"
                    )
                    try:
                        with patch(target, side_effect=fail):
                            with self.assertRaises(KeyboardInterrupt if phase == "interrupt" else VoiceAuthError):
                                create_voice_auth_key_file(self.path, key=bytes(range(32)))
                        if replace_file:
                            self.assertEqual(self.path.read_bytes(), b"replacement")
                        else:
                            self.assertFalse(self.path.exists())
                    finally:
                        self.path.unlink(missing_ok=True)

    def test_key_creation_rejects_replacement_around_hardening(self) -> None:
        for phase in ("fsync", "harden"):
            with self.subTest(phase=phase):
                original_fsync = os.fsync
                def substitute(argument):
                    if phase == "fsync":
                        original_fsync(argument)
                    self.path.unlink()
                    self.path.write_bytes(b"replacement")
                target = "jarvis_voice.voice_auth." + (
                    "os.fsync" if phase == "fsync" else "_harden_windows_key_file"
                )
                try:
                    with patch(target, side_effect=substitute):
                        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_file_invalid"):
                            create_voice_auth_key_file(self.path, key=bytes(range(32)))
                    self.assertEqual(self.path.read_bytes(), b"replacement")
                finally:
                    self.path.unlink(missing_ok=True)

    def test_key_json_rejects_trailing_data_after_control_z(self) -> None:
        create_voice_auth_key_file(self.path, key=bytes(range(32)))
        with self.path.open("ab") as output:
            output.write(b"\x1a trailing data")
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_format_invalid"):
            load_voice_auth_key(self.path)

    def test_windows_acl_rejects_foreign_and_unsupported_grants(self) -> None:
        descriptor = Mock()
        descriptor.GetSecurityDescriptorOwner.return_value = "user"
        acl = descriptor.GetSecurityDescriptorDacl.return_value
        acl.GetAceCount.return_value = 1
        security = SimpleNamespace(
            ACCESS_ALLOWED_ACE_TYPE=0, ACCESS_ALLOWED_OBJECT_ACE_TYPE=5,
            SE_FILE_OBJECT=1, OWNER_SECURITY_INFORMATION=1, DACL_SECURITY_INFORMATION=4,
            GetNamedSecurityInfo=lambda *args: descriptor,
        )
        modules = {
            "win32con": SimpleNamespace(FILE_ATTRIBUTE_REPARSE_POINT=1024),
            "win32file": SimpleNamespace(GetFileAttributes=lambda path: 0),
        }
        with patch.dict(sys.modules, modules), patch.object(voice_auth, "os", SimpleNamespace(name="nt")), patch.object(voice_auth, "_windows_key_sids", return_value=(security, "user", "system", "administrators")):
            for ace_type in (0, 5):
                for sid in ("user", "system", "administrators", "foreign"):
                    with self.subTest(ace_type=ace_type, sid=sid):
                        acl.GetAce.return_value = ((ace_type, 0), 1, sid) if ace_type == 0 else ((ace_type, 0), 1, None, None, sid)
                        if sid == "foreign":
                            with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_permissions_invalid"):
                                voice_auth._validate_windows_key_permissions(self.path)
                        else:
                            voice_auth._validate_windows_key_permissions(self.path)
            acl.GetAce.side_effect = NotImplementedError("unsupported ACE")
            with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_permissions_invalid"):
                voice_auth._validate_windows_key_permissions(self.path)

    @unittest.skipUnless(os.name == "posix", "POSIX key mutation fixtures")
    def test_key_reader_rejects_changes_during_read(self) -> None:
        for mutation in ("replace", "content", "permissions", "symlink"):
            with self.subTest(mutation=mutation):
                create_voice_auth_key_file(self.path, key=bytes(range(32)))
                original = self.path.read_bytes()
                metadata = self.path.stat()
                real_read = os.read
                changed = False
                def change_after_read(descriptor, maximum):
                    nonlocal changed
                    content = real_read(descriptor, maximum)
                    if content and not changed:
                        changed = True
                        if mutation == "replace":
                            self.path.unlink()
                            self.write(original)
                        elif mutation == "content":
                            self.path.write_bytes(original.replace(b'"key_version":1', b'"key_version":2'))
                            os.utime(self.path, ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000))
                        elif mutation == "permissions":
                            self.path.chmod(0o640)
                        else:
                            target = self.root / "original.json"
                            self.path.rename(target)
                            self.path.symlink_to(target)
                    return content
                try:
                    with patch.object(os, "read", side_effect=change_after_read):
                        with self.assertRaises(VoiceAuthError):
                            load_voice_auth_key(self.path)
                finally:
                    self.path.unlink(missing_ok=True)

    def test_windows_key_reader_checks_the_opened_handle_acl(self) -> None:
        create_voice_auth_key_file(self.path, key=bytes(range(32)))
        def security_descriptor(sid):
            acl = SimpleNamespace(GetAceCount=lambda: 1, GetAce=lambda index: ((0, 0), 1, sid))
            return SimpleNamespace(GetSecurityDescriptorOwner=lambda: "user", GetSecurityDescriptorDacl=lambda: acl)
        for change_during_read in (False, True):
            with self.subTest(change_during_read=change_during_read):
                exposed = not change_during_read
                real_read = os.read
                def read(descriptor, maximum):
                    nonlocal exposed
                    data = real_read(descriptor, maximum)
                    exposed = True
                    return data
                security = SimpleNamespace(
                    ACCESS_ALLOWED_ACE_TYPE=0, ACCESS_ALLOWED_OBJECT_ACE_TYPE=5,
                    SE_FILE_OBJECT=1, OWNER_SECURITY_INFORMATION=1, DACL_SECURITY_INFORMATION=4,
                    GetNamedSecurityInfo=lambda *args: security_descriptor("user"),
                    GetSecurityInfo=lambda *args: security_descriptor("foreign" if exposed else "user"),
                )
                def fstat(descriptor):
                    metadata = os.fstat(descriptor)
                    return SimpleNamespace(**{**{name: getattr(metadata, name) for name in dir(metadata) if name.startswith("st_")}, "st_file_attributes": 0})
                fake_os = SimpleNamespace(**{**vars(os), "name": "nt", "read": Mock(side_effect=read), "fstat": fstat})
                modules = {
                    "win32con": SimpleNamespace(FILE_ATTRIBUTE_REPARSE_POINT=1024),
                    "win32file": SimpleNamespace(GetFileAttributes=lambda path: 0),
                    "msvcrt": SimpleNamespace(get_osfhandle=lambda descriptor: descriptor),
                }
                with patch.dict(sys.modules, modules), patch.object(voice_auth, "os", fake_os), patch.object(voice_auth, "_windows_key_sids", return_value=(security, "user", "system", "administrators")):
                    with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_permissions_invalid"):
                        load_voice_auth_key(self.path)
                if change_during_read:
                    fake_os.read.assert_called()
                else:
                    fake_os.read.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX creation checkpoint")
    def test_key_creation_never_hardens_an_observed_replacement(self) -> None:
        original_fchmod = os.fchmod
        def replace_after_chmod(descriptor, mode):
            original_fchmod(descriptor, mode)
            self.path.unlink()
            self.path.write_bytes(b"replacement")
        with patch.object(os, "fchmod", side_effect=replace_after_chmod), patch.object(voice_auth, "_harden_windows_key_file") as harden:
            with self.assertRaises(VoiceAuthError):
                create_voice_auth_key_file(self.path, key=bytes(range(32)))
        harden.assert_not_called()
        self.assertEqual(self.path.read_bytes(), b"replacement")

    def test_key_creation_checks_private_handle_before_writing(self) -> None:
        def reject(path, *, descriptor=None):
            if descriptor is not None:
                self.assertEqual(os.fstat(descriptor).st_size, 0)
                raise VoiceAuthError("voice_auth.key_permissions_invalid")
        with patch.object(voice_auth, "_validate_windows_key_permissions", side_effect=reject):
            with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_permissions_invalid"):
                create_voice_auth_key_file(self.path, key=bytes(range(32)))
        self.assertFalse(self.path.exists())

    def test_windows_creation_validates_private_parent_before_writing(self) -> None:
        with (
            patch(
                "jarvis_voice.voice_auth._validate_key_parent_security"
            ) as validate_creation_parent,
        ):
            create_voice_auth_key_file(self.path, key=bytes(range(32)))
        validate_creation_parent.assert_called_once_with(self.root)

        with patch(
            "jarvis_voice.voice_auth._validate_windows_key_permissions"
        ) as validate_windows_parent:
            voice_auth._validate_key_parent_security(
                self.root,
                platform_name="nt",
            )
        validate_windows_parent.assert_called_once_with(self.root)

    def test_relative_missing_directory_and_symlink_paths_fail_closed(self) -> None:
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_path_invalid"):
            create_voice_auth_key_file("relative.json")
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_path_invalid"):
            create_voice_auth_key_file(self.root / "missing" / "key.json")
        target = self.root / "target.json"
        create_voice_auth_key_file(target, key=bytes(range(32)))
        link = self.root / "link.json"
        link.symlink_to(target)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_file_invalid"):
            load_voice_auth_key(link)

    @unittest.skipUnless(os.name == "posix", "POSIX owner/mode contract")
    def test_group_or_world_permissions_are_rejected(self) -> None:
        create_voice_auth_key_file(self.path, key=bytes(range(32)))
        self.path.chmod(0o640)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_permissions_invalid"):
            load_voice_auth_key(self.path)

    def test_key_json_requires_exact_fields_schema_version_and_canonical_base64(self) -> None:
        valid = {
            "schema_version": KEY_FILE_SCHEMA,
            "key_version": 1,
            "key_b64": "AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8",
        }
        variants = []
        extra = dict(valid); extra["extra"] = True; variants.append(json.dumps(extra).encode())
        wrong_schema = dict(valid); wrong_schema["schema_version"] = "jarvis.voice-auth-key/2"; variants.append(json.dumps(wrong_schema).encode())
        wrong_version = dict(valid); wrong_version["key_version"] = True; variants.append(json.dumps(wrong_version).encode())
        bad_key = dict(valid); bad_key["key_b64"] = "A" * 42 + "="; variants.append(json.dumps(bad_key).encode())
        variants.append(
            b'{"schema_version":"jarvis.voice-auth-key/1","key_version":1,'
            b'"key_version":2,"key_b64":"AAECAwQFBgcICQoLDA0ODxAREhMUFRYXGBkaGxwdHh8"}'
        )
        for index, raw in enumerate(variants):
            with self.subTest(index=index):
                self.write(raw)
                with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_format_invalid"):
                    load_voice_auth_key(self.path)
                self.path.unlink()

    def test_key_file_must_be_bounded_regular_and_nonempty(self) -> None:
        self.write(b"")
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_format_invalid"):
            load_voice_auth_key(self.path)
        self.path.unlink()
        self.write(b"x" * 1_025)
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_format_invalid"):
            load_voice_auth_key(self.path)
        self.path.unlink()
        directory = self.root / "directory"
        directory.mkdir()
        with self.assertRaisesRegex(VoiceAuthError, "voice_auth.key_file_invalid"):
            load_voice_auth_key(directory)

    def test_key_errors_are_content_free(self) -> None:
        sensitive = self.root / "private-user-token-name.json"
        try:
            load_voice_auth_key(sensitive)
        except VoiceAuthError as exc:
            self.assertEqual(str(exc), "voice_auth.key_unavailable")
            self.assertNotIn(str(sensitive), str(exc))
        else:  # pragma: no cover - assertion clarity
            self.fail("missing key file was accepted")


if __name__ == "__main__":
    unittest.main()
