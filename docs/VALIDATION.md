# Windows validation — 2026-10-12

All nine catalog variants were built for Windows x64 from the Hugging Face
cache, with embedded weights in both the executable and shared library. The
final rebuild used `--offline`; it did not read sibling model repositories.
Portable directories are in `dist/`. A CLAP INT4 ZIP is also available there.
Linux execution and WSL setup were deferred at the owner's request.

Sixteen Rust tests, three Python build-contract tests and `cargo fmt --check`
passed. Package checks exercise the CLI with a restricted PATH, UTF-8 audio
paths, C ABI file/PCM embedding, repeatability, shape/mask accessors, error
handling and unchanged source audio. This is validation on the current Windows
host; a separate clean Windows machine has not been exercised.

The reference comparison uses the same selected ONNX graph with pinned Python
processors and ONNX Runtime. Gates remain max absolute error below 0.0005 for
FP32 or 0.005 for quantized outputs, and cosine similarity above 0.9999.
Failures are saved in JSON and make `smoke.py --all --reference` fail after
checking every package. Native runtime checks and reference parity are reported
separately.

Synth quality uses ten fixed ordered triples across pitch, harmonic mix, filter
cutoff, decay and FM depth. Both adjacent distances must be at most the endpoint
distance. At least nine passing triples are required by the default automated
expectation. The CLI reports observations; `synth_checks.py` enforces the gate.
The values and thresholds have not been adjusted to hide violations.

| Variant | CLI/C ABI | Reference max error | Reference cosine | Reference gate | Synth triples | Quality gate |
|---|---|---:|---:|---|---:|---|
| CLAP FP32 | Pass | 5.96e-8 | 1.000000000 | Pass | 9/10 | Pass |
| CLAP INT8 | Pass | 0 | 1.000000000 | Pass | 9/10 | Pass |
| CLAP INT4 | Pass | 0.000953 | 0.999983435 | Pass | 9/10 | Pass |
| Gemma final FP32 | Pass | 1.38e-7 | 1.000000000 | Pass | 9/10 | Pass |
| Gemma final INT8 | Pass | 0 | 1.000000000 | Pass | 6/10 | Fail |
| Gemma final INT4 | Pass | 0.005978 | 0.999175417 | Fail | 8/10 | Fail |
| Gemma tower FP32 | Pass | 0.000133 | 1.000000000 | Pass | 9/10 | Pass |
| Gemma tower INT8 | Pass | 0 | 1.000000000 | Pass | 6/10 | Fail |
| Gemma tower INT4 | Pass | 1.822412 | 0.999711274 | Fail | 8/10 | Fail |

Full-precision measurements and individual violations are in
`test-results/<model>-smoke.json` and `<model>-synth.json`. Reference success on
this smoke signal does not imply perceptual quality: INT8 Gemma is an example.

## Quantization diagnostic

On a one-second 220 Hz signal, the native Gemma features differed from the
pinned processor by at most 4.7683716e-7 (one float32 step), with 526 differing
values out of 12,672. Feeding those native features directly to Python ONNX
Runtime reproduced the INT4 final-embedding drift: max absolute error
0.0059779976 and cosine similarity 0.9991754168. Perturbing the processor's own
features by one float32 step also produced drift (max error 0.0060280897,
cosine 0.9989478733). These observations indicate amplification of tiny frontend
rounding differences in this quantized artifact; they do not establish parity
for arbitrary audio. `examples/gemma_frontend.rs` provides a native feature
dump for further investigation.

Gemma FP32 final/tower outputs and all quantized variants remain distinct model
spaces. Raw towers expose the unpooled state tensor and mask as requested.
Human listening is still required before choosing an app default.

## Artifact repair

The Gemma FP32 final graph really did reference twelve small external Constant
tensors. They are now inline in the graph, leaving only `audio_embedding.onnx`
and the existing `audio_embedding.data`; the large weight file was unchanged.
The exporter performs this consolidation before packing initializer data.
Signed repair commit: [33a04f9](https://huggingface.co/bdrgon/embeddinggemma2-audio-tower-onnx/commit/33a04f9fcc02ff7bb439bf30c62dbcc675fab214).
The supplied signing key worked. Its temporary imported signing home was
removed; the original supplied file remains in the workspace.
