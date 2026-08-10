# Manifest v1 exact shapes

All objects reject unknown fields. All artifact objects are exactly
`{"path":string,"size_bytes":integer,"sha256":64-lowercase-hex}`.
All arrays that carry artifact/package identities are deterministically sorted.
Example hashes shown below are placeholders, not qualified identities.

The top-level object is exactly:

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

Apple Silicon `backend` is exactly:

```json
{"type":"mlx-worker","worker":{"path":"runtime/mlx_worker.py","size_bytes":0,"sha256":"..."}}
```

Apple Silicon `python` is exactly:

```json
{
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
