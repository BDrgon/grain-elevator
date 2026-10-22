# Grain Elevator

Native audio -> model-specific vectors/tensors. Rust API, C ABI and CLI, with
build-selected Hugging Face weights embedded inside the executable and library.
No index or app UI. Current contract: [docs/SPEC.md](docs/SPEC.md).

## Build on Windows

Install Rust's Windows GNU toolchain and MinGW, plus a development Python:

```powershell
python -m pip install -r tools/requirements.txt
python tools/build.py --model clap-fp32 --platform windows-x64 --vc-redist C:/licensed/x64/CRT
python tools/build.py --model all --platform windows-x64 --vc-redist C:/licensed/x64/CRT
python tools/build.py --model https://huggingface.co/bdrgon/embeddinggemma2-audio-tower-onnx/resolve/main/audio_embedding_int8.onnx --platform windows-x64 --vc-redist C:/licensed/x64/CRT
```

`--model` is a catalog key, `all`, or a registered artifact's `resolve`/`blob` URL.
`--models OTHER.json` selects another catalog. All nine variants are registered
in [models.json](models.json). Add new model contracts there and implement the
frontend when needed; an arbitrary ONNX graph is not automatically compatible.

`--offline` uses cached files and the catalog's revision pin (including URL `main`).
`--cache-dir DIR` adds a build cache after the standard Hugging Face cache.
Catalog builds always use the pinned commit/checksums; online URL builds resolve
the requested revision to a commit and its checksums. Authentication, when needed,
uses the standard HF token setup. Adjacent model repos are never read.

`--vc-redist` contains Microsoft's redistributable x64 CRT DLLs and their notices.
The package copies these for app-local operation without an installer. An explicit
`--allow-system-crt` creates a development package depending on the installed CRT.
`--no-archive` leaves the portable directory only. Normal builds produce ZIPs with
both CLI and shared library, embedded weights, ONNX Runtime, notices and checksums.
The largest FP32 models need several GB of disk/RAM; weights are duplicated across
the independently usable executable and library.

## Run

```powershell
./dist/clap-fp32-windows-x64/grain-elevator.exe info
./dist/clap-fp32-windows-x64/grain-elevator.exe embed sample.wav
./dist/clap-fp32-windows-x64/grain-elevator.exe embed sample.wav --output vector.f32
./dist/clap-fp32-windows-x64/grain-elevator.exe synth-test --report distances.json
```

The output file is float32 little-endian with a `.metadata.json` companion naming
the model identity, tensor shape and mask. File inputs remain unchanged. Output
files must be new paths; the CLI refuses to overwrite existing files. CLAP uses
the first 10 seconds; Gemma uses the first 30. Final embeddings have 512 values.
Gemma tower outputs are raw states with shape `[1,tokens,1536]` and duration-dependent
length; no pooling is applied. Their mask travels with the result.

## API

```rust,no_run
use grain_elevator::Encoder;
let encoder = Encoder::open_image("dist/clap-fp32-windows-x64/grain_elevator.dll", None)?;
let value = encoder.embed_file("sample.wav")?;
let pcm_value = encoder.embed_pcm(&mono_samples, 48_000)?;
// value.values, value.shape, value.mask, value.identity
```

`Encoder::open()` loads the payload embedded in the current executable. Rust apps
can use `open_image` with a packaged CLI/shared library. C callers use
[include/grain_elevator.h](include/grain_elevator.h): `ge_open`, `ge_embed_file` /
`ge_embed_pcm`, data/shape/mask/identity accessors and explicit handle release.
Provide valid buffers/handles. `GRAIN_ORT_LIB` can explicitly locate ONNX Runtime
for a host app; packaged CLI builds find it beside their executable.

## Verify

```powershell
cargo test --locked
python tools/smoke.py --all
python tools/smoke.py dist/clap-fp32-windows-x64 --reference
python tools/synth_checks.py dist/clap-fp32-windows-x64
python tools/test_build.py
```

Reference comparison additionally needs the pinned Transformers 5.19.0, torch
2.8.0 and processors in the development environment. Checked-in Gemma frontend
fixtures come from that processor. Python is not shipped with the product.
See [docs/VALIDATION.md](docs/VALIDATION.md) for measured Windows results and
model quality/reference failures retained by this initial implementation.

Synth measurements report both adjacent-versus-span inequalities, violations and
the pass fraction. `synth_checks.py` fails below 90% passing triples by default;
`--min-pass-fraction` makes this initial 'almost always' threshold explicit and
configurable. Human listening remains the final quality gate. All sweep clips
have equal duration.

## Later Linux build

On a supplied Linux x64 machine, install the same development Python requirements
and Rust GNU target, then run:

```sh
sh tools/build-linux.sh --model all
python3 tools/smoke.py --all
```

Produces portable directories/tarballs bundling `libonnxruntime.so`. Linux uses
the build host's glibc baseline; build on the oldest supported distribution and
verify there. Linux execution is deferred until a machine is available. The CI
workflow currently builds Windows only; it has not been dispatched here.

Dependencies: Symphonia (unmodified MPL-2.0), ONNX Runtime (MIT), Rust dependency
licenses and selected model notices are included in each portable package.
