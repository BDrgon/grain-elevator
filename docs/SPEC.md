# Grain Elevator initial implementation

Owner scope: native audio-to-embedding library only. Indexing belongs to the
future Silo component; browsing/playback belong to the app.

## Model selection and distribution

`models.json` pins nine artifacts: CLAP FP32/INT8/INT4; Gemma final embedding
FP32/INT8/INT4; Gemma raw audio tower FP32/INT8/INT4. Build selection accepts a
catalog key, `all`, or a registered artifact's public Hugging Face URL. URLs
resolve to a commit before downloading. The standard HF cache is checked first,
then the selected build cache. Adjacent model-repo files are never used.

Validate checksums, graph I/O and every external-data reference, including
Constant attributes. Embed graph and weight sidecars inside each packaged
executable/shared library. The embedded overlay avoids huge compiler arrays and
linker object-size limits; it is still part of the executable, not a runtime
download or weights directory. Runtime verifies payload hashes and feeds the
embedded external weights directly into ONNX Runtime memory.

Windows x64 is the initial built/tested platform. Linux x64 native build scripts
are included for a later supplied machine. `platforms.json` describes targets,
runtime filenames and packaging. New architectures require the corresponding
Rust target, ONNX Runtime build and native verification, not just a renamed ZIP.

## Audio and output contracts

Read-only file/byte decoding and mono PCM inputs. No Python, Hub calls, telemetry,
UI, index or model export at runtime. ONNX Runtime is bundled next to the binary.
The public surfaces are Rust and an owned-handle C ABI; the CLI is diagnostic.

CLAP: first 10 seconds, 48 kHz mono, original repeat-then-zero-pad log-mel
frontend, 1001x64 features. The graph returns normalized `[1,512]` embeddings.

Gemma: first 30 seconds, 16 kHz mono, pinned Gemma4 frontend, periodic 320-sample
Hann window, 160-sample hop, 512 FFT, magnitude/HTK 128-bin mel, natural log with
0.001 floor and semicausal padding. Waveform padding rounds to 128 samples;
invalid feature frames are zero and masked. Audio-only token layout is fixed to
the pinned processor. Empty/non-finite input and audio with no valid tokens fail.

Gemma final graphs return normalized `[1,512]` vectors. As requested, the tower
graphs return the **unaltered raw `[1,audio_tokens,1536]` state tensor plus its
mask**. Values are exposed contiguously with shape; N is duration-dependent.
No pooling or normalization is applied to tower outputs. They do not share an
embedding space with final embeddings. Compare raw tensors only when identities,
shapes and masks match; cross-duration similarity is not defined in this release.

## Correctness and synth measurements

Require pinned processor frontend parity, graph-output parity, repeatability,
finite outputs, expected shape, error handling, portable CLI/C ABI execution and
unchanged source audio. Identity contains repository, commit, filename, frontend
and output treatment. Never compare vectors from different identities.

Quality suite: deterministic one-second synth signals, fixed envelope/filter/
oscillator inputs except for one knob. Sweep pitch, harmonic mix, low-pass cutoff,
decay and FM depth. For each ordered triple, measure cosine distances d12, d23,
d13 and independently report d12<=d13 and d23<=d13, with 1e-6 numeric tolerance.
Each suite reports violation cases and its passing fraction. All three clips
have equal duration, including tower measurements.

The automated expectation is that at least 90% of frozen triples pass, an initial
explicit interpretation of "almost always" configurable in `synth_checks.py`.
Learned similarity need
not be monotonic for every knob/range: do not alter test values or relax thresholds
to hide violations. Measurements do not replace human listening. A strict runner
requires its selected minimum fraction; the default CLI reports measurements
without converting a perceptual observation into an implementation error.
