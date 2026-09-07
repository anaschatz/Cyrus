# Manifest v1 exact shapes

All objects reject unknown fields. All artifact objects are exactly
`{"path":string,"size_bytes":integer,"sha256":64-lowercase-hex}`.
All arrays that carry artifact/package identities are deterministically sorted.
Example hashes shown below are placeholders, not qualified identities.
The manifest itself is opened as one bounded regular file with no leaf symlink
or Windows reparse point. On Windows its absolute spelling must use a drive
letter (`C:\...`); UNC, device-namespace and drive-relative paths are rejected
before resolution, and every existing ancestor is inspected for a reparse
point. File identity, size and modification state must remain stable across the
read. Direct `serve`, `qualify`, and verification calls share this admission
path.

The frozen Windows manifest builder requires the same absolute drive-letter
form and applies the pre-resolution ancestor check to the runtime root, output,
runtime/model directories, executable, model and DLLs. It publishes a complete
temporary file by an exclusive same-directory link and refuses an existing
destination, including one created by a racing process. Re-materializing a
different identity therefore uses a fresh output path; the builder never
overwrites a sealed manifest.

The top-level object has the following required fields. `prompt_profile` and
`profile_qualification_report` are the only optional top-level fields. They
are accepted only together for the exact Windows Gemma primary; manifests
that omit both retain the legacy renderer path.

```json
{
  "manifest_schema_version": 1,
  "alias": "jarvis-local",
  "platform": "windows-x86_64",
  "backend": {},
  "model": {},
  "python": {},
  "capabilities": {
    "text": true,
    "stream": true,
    "tools": false,
    "json_object": false,
    "json_schema": false
  },
  "qualification_report": "qualification-report.json"
}
```

The selected Windows Gemma primary adds
`"profile_qualification_report":"qualification-profile-local-primary.json"`
and this exact `prompt_profile` object:

```json
{
  "id": "google-gemma4-12b-it-qat-q4_0-v1",
  "model_repository": "google/gemma-4-12B-it-qat-q4_0-gguf",
  "model_revision": "29d097773436b69ff9feafd636ab4cf873786537",
  "model_artifact": {
    "filename": "gemma-4-12b-it-qat-q4_0.gguf",
    "size_bytes": 6975879296,
    "sha256": "93567e57a8fe10b23569b9d9ec38cd005deedf71e29477c421a4b83f418a538b"
  },
  "embedded_chat_template": {
    "metadata_key": "tokenizer.chat_template",
    "size_bytes": 18683,
    "sha256": "ae53464bf3be25802b3a5b37def7fd89667067d7577049b3b2d74c4d8de4c6d4"
  },
  "renderer": {
    "id": "cyrus.gemma4-appliance-renderer/1",
    "sha256": "3ce59b2dadfeeff369b337da785deb096ca50d8a2182570cc9db36f7f63a1951"
  },
  "tool_parser": {
    "id": "cyrus.gemma4-single-tool-parser/1",
    "sha256": "e118a2ab2d39113fd2a18964f8f6ade87692afeed26c87d95d87a0dcb5c6afd4"
  },
  "thinking": false
}
```

The profile is valid only with the one exact model filename, size, and hash
shown above. Its template identity comes from `tokenizer.chat_template`
embedded in that hash-locked GGUF. Serving does not fetch, load, or interpret
Jinja; it selects the appliance-owned renderer and parser only when this whole
profile is present and exact. Existing Windows and Mac manifests without the
field keep their prior behavior only for text/stream capability. A Windows
manifest that advertises `tools`, `json_object`, or `json_schema` without the
exact prompt profile is rejected during manifest admission; it cannot fall
back to the generic renderer.

The generic `qualification_report` proves the appliance protocol, isolation,
cancellation and native platform gates. It is necessary but does not make the
Gemma profile ready. Production additionally requires the declared profile
report and its preserved sanitized evidence. Both must bind the exact generic
report bytes/evidence/runner, manifest and runtime identities, model,
renderer/parser, tokenizer, quality/performance fixtures, ASR assets and power
profile. A missing, failed, copied, stale, identity-mismatched or modified
profile report keeps `/readyz` false. Mac and legacy text-only manifests do not
gain this second-report requirement.

Every accepted primary profile report and its preserved evidence must declare
`"attestation_level":"identity-bound-local-operator"` and
`"operator_trust_required":true`. `verify-readiness` returns the same two
fields. This is an explicit trust disclosure, not a cryptographic attestation:
the current checks detect stale, mismatched, corrupt, and independently edited
artifacts, but a malicious process or operator with the same local account can
coordinate new JSON and hashes. Protecting against that actor requires a later
hardware-backed signing/attestation design and is outside v1.

Windows `backend` is exactly:

```json
{
  "type": "llama-server",
  "llama_cpp_commit": "0000000000000000000000000000000000000000",
  "build_id": "exact-build-identifier",
  "executable": {"path":"runtime/llama-server.exe","size_bytes":0,"sha256":"..."},
  "dlls": [],
  "parameters": {
    "context_window": 32768,
    "gpu_layers": 0,
    "batch_size": 2048,
    "ubatch_size": 512,
    "threads": 8,
    "flash_attention": true
  }
}
```

The parameter keys are optional but no other key is accepted. `dlls` is sorted
by case-insensitive basename and must equal every DLL next to the executable.
Windows `python` is exactly `{}`.

The managed child is launched by absolute executable path with its verified
executable directory as the working directory. Its environment contains no
inherited `PATH`; only Windows system-root/temp values and the fresh private
backend bearer are forwarded. Consequently, an undeclared DLL cannot be
silently supplied by the user's shell or current working directory. Any native
dependency required by the pinned build must live in the manifest-verified
runtime bundle (or be a Windows system component), otherwise startup fails
closed.

Apple Silicon `backend` is exactly:

```json
{"type":"mlx-worker","worker":{"path":"runtime/mlx_worker.py","size_bytes":0,"sha256":"..."}}
```

Apple Silicon `python` has the following shape:

```json
{
  "environment_sha256": "...",
  "implementation": "CPython",
  "version": "3.11.0",
  "executable": {"path":"python/bin/python3.11","size_bytes":0,"sha256":"..."},
  "site_packages": "python/lib/python3.11/site-packages",
  "packages": [
    {
      "name": "canonical-package-name",
      "version": "1.0.0",
      "wheel_filename": "canonical_package_name-1.0.0-py3-none-any.whl",
      "wheel": {"path":"wheels/canonical_package_name-1.0.0-py3-none-any.whl","size_bytes":0,"sha256":"..."}
    }
  ]
}
```

`packages` is sorted and contains every transitive installed distribution.
`environment_sha256` is a lowercase 64-character SHA-256 digest required for
runtime verification and qualified cloning. It binds every regular file below
the Python installation by relative path, size, content hash and executable-bit
presence, including the standard library and generated command wrappers.
The builder computes it using `python_environment.hash_environment`; it is not
the hash of the interpreter alone. Only the nine exact CPython convenience
aliases accepted by that module are allowed and excluded from the digest.

Legacy manifests without this field can be parsed for inspection but cannot
pass verification or qualified cloning. Rebuild a clean installation, generate
a new manifest and obtain fresh native qualification. Run the sealed Python
with `-B` to avoid creating bytecode files that change its identity. Installed
wheel contents and RECORD entries are checked separately from this digest.

`model` is exactly:

```json
{
  "format": "gguf",
  "context_window": 32768,
  "artifacts": [
    {"path":"models/model.gguf","size_bytes":0,"sha256":"..."}
  ]
}
```

For Apple Silicon, `format` is `mlx` and `artifacts` enumerates every weight,
tokenizer, configuration, chat-template, and adapter file.

Mac model artifacts must share one dedicated top-level model directory. At
verification time every regular file below that directory must be declared;
extra files, missing files, symlinks, or reparse points fail identity.

Manifest and public JSON digests use RFC 8785 over the supported I-JSON domain:
UTF-16 code-unit member ordering, shortest ECMAScript binary64 rendering,
finite numbers only, and integers limited to `[-9007199254740991,
9007199254740991]`.
