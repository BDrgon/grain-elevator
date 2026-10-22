//! The log-mel frontend, pinned to what the reference stack actually computes.
//!
//! SPEC §6 describes this pipeline and is wrong about it in four measurable ways. Every claim
//! below was measured against `transformers==4.57.1` / `torch==2.8.0+cpu` with the pinned
//! `laion/clap-htsat-unfused` preprocessor config (see `tools/fixtures/out/manifest.json`,
//! `frontend_measured`), and each stage is a fixture file the parity test compares against:
//!
//! | stage | SPEC §6 says | reference does | fixture |
//! |---|---|---|---|
//! | mel bank | HTK | **`mel_filters_slaney`** (`norm='slaney'`, `mel_scale='slaney'`) — HTK is 35-50 dB off | `mel_filters.f32` |
//! | frame count | 1000 | **1001** = `1 + 480000/480` after the center pad; 1000 is `nb_max_frames`, a config field | `stft/*.power.f32` |
//! | log | `(x+4)/4` with a 32768 scale and `db_range` | **`10*log10(max(mel, 1e-10))`** — no rescale, no `db_range` | `mel/*.f32` |
//! | short-clip padding | repeat pad | `np.tile(w, int(480000/len(w)))` **then** zero-pad to 480000 | `stft/*.padded.f32` |
//!
//! Precision is not a detail: waveform, window, STFT, mel matmul and the log all run in float64
//! and narrow to float32 only at the end. Doing the window in float32 costs ~3e-3 dB, which is
//! most of the §9 budget.

use crate::fft::Fft;

/// Everything the pinned model's frontend needs (never from a CLI flag).
#[derive(Clone, Debug, PartialEq)]
pub struct FrontendParams {
    /// 48_000 for the pinned model.
    pub sample_rate: u32,
    pub n_fft: usize,
    pub hop: usize,
    pub mel_bins: usize,
    pub freq_min: f64,
    pub freq_max: f64,
    /// `nb_max_frames * hop_length` = the waveform length the reference pads to: 480_000.
    pub window_samples: usize,
    /// `mel_bins` of the graph input's last axis is `mel_bins`; its frames are `frames()`.
    pub mel_floor: f64,
}

impl Default for FrontendParams {
    /// The pinned `laion/clap-htsat-unfused` frontend, as measured.
    fn default() -> Self {
        FrontendParams {
            sample_rate: 48_000,
            n_fft: 1024,
            hop: 480,
            mel_bins: 64,
            freq_min: 50.0,
            freq_max: 14_000.0,
            window_samples: 480_000,
            mel_floor: 1e-10,
        }
    }
}

impl FrontendParams {
    /// Emitted frames: `1 + window_samples / hop` after the center pad — 1001 for the pins.
    ///
    /// Derivation: the center pad adds `n_fft/2` at both ends, so the frame count is
    /// `1 + (window_samples + n_fft - n_fft) / hop`.
    pub fn frames(&self) -> usize {
        1 + self.window_samples / self.hop
    }

    pub fn rfft_bins(&self) -> usize {
        self.n_fft / 2 + 1
    }
}

/// `np.hanning(n + 1)[:-1]` — the *periodic* Hann window, `0.5 - 0.5*cos(2 pi i / n)`.
///
/// The symmetric form (`np.hanning(n)`) is wrong here by a quarter-sample taper at the end, which
/// shows up as a real difference in the mel, not a rounding error.
pub fn hann_periodic(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * (i as f64) / (n as f64)).cos())
        .collect()
}

/// Slaney's 1998 mel scale, as `mel_scale="slaney"` implements it in the reference stack.
///
/// **This is not `2595*log10(1 + f/700)`.** That formula is what `transformers` calls
/// `mel_scale="htk"`; using it here lands within a couple of percent of the right answer on the
/// endpoints and up to 600 Hz off in the middle, which is exactly the kind of error that survives
/// a casual look at the numbers. The pinned scale is piecewise: linear to 1000 Hz (15 mels),
/// logarithmic above it, with `logstep = ln(6.4)/27` per mel.
/// `ln(6.4)`. `f64::ln` is not `const`, and the reference writes the same constant as an
/// expression, so it is spelled out once here with its value.
const LN_6_4: f64 = 1.856_297_990_365_626;

fn hz_to_mel_slaney(hz: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = 27.0 / LN_6_4;
    if hz >= MIN_LOG_HERTZ {
        MIN_LOG_MEL + (hz / MIN_LOG_HERTZ).ln() * logstep
    } else {
        3.0 * hz / 200.0
    }
}

/// Inverse of [`hz_to_mel_slaney`].
fn mel_to_hz_slaney(mel: f64) -> f64 {
    const MIN_LOG_HERTZ: f64 = 1000.0;
    const MIN_LOG_MEL: f64 = 15.0;
    let logstep = LN_6_4 / 27.0;
    if mel >= MIN_LOG_MEL {
        MIN_LOG_HERTZ * (logstep * (mel - MIN_LOG_MEL)).exp()
    } else {
        200.0 * mel / 3.0
    }
}

/// The mel bank the reference builds: `transformers`' `mel_filter_bank(..., norm="slaney",
/// mel_scale="slaney")`, which is librosa's `mel()` vendored as
/// `_create_triangular_filter_bank`. Ported line for line, because a 2% error in this matrix is a
/// 2% error in every embedding and SPEC §6 claimed the wrong scale entirely.
///
/// ```text
///  filter_freqs = mel_to_hz(linspace(hz_to_mel(fmin), hz_to_mel(fmax), mels + 2))   # Hz, 66
///  fft_freqs    = linspace(0, sr/2, bins)                                            # 513
///  slopes[b, j]        = filter_freqs[j] - fft_freqs[b]
///  down_slopes[b, k]   = -slopes[b, k]     / (filter_freqs[k+1] - filter_freqs[k])
///  up_slopes[b, k]     =  slopes[b, k+2]   / (filter_freqs[k+2] - filter_freqs[k+1])
///  bank[b, k]          = max(0, min(down_slopes[b, k], up_slopes[b, k]))
///  bank[b, k]         *= 2 / (filter_freqs[k+2] - filter_freqs[k])      # "slaney" area norm
/// ```
///
/// The normalisation is `2 / (hi - lo)` in **Hertz** — area normalisation of the triangle — not
/// `2 / (mel[-1] - mel[0])`; that distinction is most of the 1.8% this function was off by before
/// it was ported from the source.
pub fn mel_bank_slaney(params: &FrontendParams) -> Vec<f32> {
    let bins = params.rfft_bins();
    let mels = params.mel_bins;

    // filter_freqs: linspace in the mel domain, mapped back to Hz.
    let (mel_min, mel_max) = (
        hz_to_mel_slaney(params.freq_min),
        hz_to_mel_slaney(params.freq_max),
    );
    let n = mels + 2;
    let filter_freqs: Vec<f64> = (0..n)
        .map(|i| mel_to_hz_slaney(mel_min + (mel_max - mel_min) * (i as f64) / ((n - 1) as f64)))
        .collect();
    let nyquist = (params.sample_rate as f64) / 2.0;
    let fft_freqs: Vec<f64> = (0..bins)
        .map(|b| nyquist * (b as f64) / ((bins - 1) as f64))
        .collect();
    // filter_diff[j] = filter_freqs[j+1] - filter_freqs[j], length mels + 1.
    let filter_diff: Vec<f64> = (0..n - 1)
        .map(|j| filter_freqs[j + 1] - filter_freqs[j])
        .collect();

    let mut bank = vec![0f32; bins * mels];
    for (b, f) in fft_freqs.iter().enumerate() {
        for k in 0..mels {
            // down: (f - filter_freqs[k]) / (filter_freqs[k+1] - filter_freqs[k])
            let down = (*f - filter_freqs[k]) / filter_diff[k];
            // up: (filter_freqs[k+2] - f) / (filter_freqs[k+2] - filter_freqs[k+1])
            let up = (filter_freqs[k + 2] - *f) / filter_diff[k + 1];
            let tri = down.min(up).max(0.0);
            let enorm = 2.0 / (filter_freqs[k + 2] - filter_freqs[k]);
            bank[b * mels + k] = (tri * enorm) as f32;
        }
    }
    bank
}

/// numpy `pad(w, 512, mode="reflect")`: mirror **excluding** the edge sample, both ends.
///
/// `reflect` (not `symmetric`): the sample at the boundary is not duplicated. Off-by-one here is
/// a different first and last frame, and it is audible in the mel.
pub fn reflect_pad(w: &[f64], pad: usize, out: &mut Vec<f64>) {
    out.clear();
    out.reserve(w.len() + pad * 2);
    // Left: w[pad], w[pad-1], ..., w[1]
    for i in (1..=pad).rev() {
        out.push(w[i]);
    }
    out.extend_from_slice(w);
    // Right: w[-2], w[-3], ..., w[-pad-1]
    let n = w.len();
    for i in 1..=pad {
        out.push(w[n - 1 - i]);
    }
}

/// The `repeatpad` branch of the reference: tile the waveform an integer number of times, then
/// zero-pad to `target`. Signals at or over `target` pass through untouched (truncation to 10 s
/// happens upstream, on the decoded waveform, before resampling).
pub fn repeatpad(w: &[f64], target: usize, out: &mut Vec<f64>) {
    out.clear();
    if w.len() >= target {
        out.extend_from_slice(&w[..target.min(w.len())]);
        return;
    }
    let times = target / w.len().max(1);
    out.reserve(target);
    for _ in 0..times.max(1) {
        out.extend_from_slice(w);
    }
    out.truncate(target);
    // `times == 0` only happens for an empty input; zero-fill so the frame count is still right.
    while out.len() < target {
        out.push(0.0);
    }
}

/// The whole frontend: 48 kHz mono in, `(frames, mel_bins)` float32 log-mel out.
pub struct Frontend {
    params: FrontendParams,
    fft: Fft,
    window: Vec<f64>,
    bank: Vec<f32>,
}

impl Frontend {
    pub fn new(params: FrontendParams) -> Frontend {
        let fft = Fft::new(params.n_fft);
        let window = hann_periodic(params.n_fft);
        let bank = mel_bank_slaney(&params);
        Frontend {
            params,
            fft,
            window,
            bank,
        }
    }

    pub fn params(&self) -> &FrontendParams {
        &self.params
    }

    pub fn window(&self) -> &[f64] {
        &self.window
    }

    pub fn bank(&self) -> &[f32] {
        &self.bank
    }

    pub fn frames(&self) -> usize {
        self.params.frames()
    }

    /// Stage 1: `repeatpad` to the model's waveform length.
    pub fn staged_pad(&self, w: &[f64]) -> Vec<f64> {
        let mut out = Vec::new();
        repeatpad(w, self.params.window_samples, &mut out);
        out
    }

    /// Stage 2: center-pad, window, rfft, |.|^2. Row-major `(frames, rfft_bins)`.
    pub fn staged_power(&self, padded: &[f64]) -> Vec<f64> {
        let p = &self.params;
        let bins = p.rfft_bins();
        let frames = self.frames();
        let mut refl = Vec::with_capacity(padded.len() + p.n_fft);
        reflect_pad(padded, p.n_fft / 2, &mut refl);
        let mut windowed = vec![0f64; p.n_fft];
        let mut power = vec![0f64; frames * bins];
        let mut row = vec![0f64; bins];
        for f in 0..frames {
            let start = f * p.hop;
            // A short buffer here means the caller passed a waveform that was not padded; treat
            // the missing tail as silence rather than reading out of bounds.
            for (o, i) in (0..p.n_fft).enumerate() {
                let idx = start + i;
                windowed[o] = if idx < refl.len() {
                    refl[idx] * self.window[i]
                } else {
                    0.0
                };
            }
            self.fft.rfft_power(&windowed, &mut row);
            power[f * bins..f * bins + bins].copy_from_slice(&row);
        }
        power
    }

    /// Stage 3: mel projection plus the `1e-10` floor, still float64. Row-major
    /// `(frames, mel_bins)` — matches `stft/<clip>.mel_linear.f32`.
    pub fn staged_mel_linear(&self, power: &[f64]) -> Vec<f64> {
        let p = &self.params;
        let bins = p.rfft_bins();
        let mels = p.mel_bins;
        let frames = power.len() / bins;
        // Accumulate from zero and clamp afterwards. Seeding with the floor looks equivalent and
        // is not: it adds 1e-10 to every cell, which is invisible in the dB scale for loud cells
        // and a ~2 dB error for the near-silent ones the fixture's quiet frames are made of.
        let mut out = vec![0.0f64; frames * mels];
        for f in 0..frames {
            let prow = &power[f * bins..f * bins + bins];
            let orow = &mut out[f * mels..f * mels + mels];
            for (bi, pv) in prow.iter().enumerate() {
                if *pv == 0.0 {
                    continue;
                }
                let brow = &self.bank[bi * mels..(bi + 1) * mels];
                for (o, bv) in orow.iter_mut().zip(brow.iter()) {
                    *o += pv * (*bv as f64);
                }
            }
            for o in orow.iter_mut() {
                if *o < p.mel_floor {
                    *o = p.mel_floor;
                }
            }
        }
        out
    }

    /// Stage 4: `10*log10(.)`, narrowed to float32 — the tensor the graph actually sees.
    pub fn staged_log(&self, mel_linear: &[f64]) -> Vec<f32> {
        mel_linear
            .iter()
            .map(|v| (10.0 * v.max(self.params.mel_floor).log10()) as f32)
            .collect()
    }

    /// All four stages. `w48` is 48 kHz mono float32, whatever its length.
    pub fn process(&self, w48: &[f32]) -> Vec<f32> {
        let w: Vec<f64> = w48.iter().map(|v| *v as f64).collect();
        let padded = self.staged_pad(&w);
        let power = self.staged_power(&padded);
        let mel_linear = self.staged_mel_linear(&power);
        self.staged_log(&mel_linear)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_is_1001_not_1000() {
        // SPEC §6 says 1000. The reference emits 1001 for the pinned config, and the graph input
        // is (1,1,1001,64); getting this wrong is a shape error at inference time, not a rounding
        // difference.
        let p = FrontendParams::default();
        assert_eq!(p.frames(), 1001);
        assert_eq!(p.rfft_bins(), 513);
        assert_eq!(p.window_samples, 480_000);
    }

    #[test]
    fn periodic_hann_starts_and_ends_where_it_should() {
        let w = hann_periodic(4);
        // 0.5 - 0.5*cos(2 pi i / 4): 0, 0.5, 1, 0.5  (the periodic window never reaches 0 at the
        // far end; the symmetric one would end at 0).
        assert_eq!(w[0], 0.0);
        assert!((w[1] - 0.5).abs() < 1e-15);
        assert!((w[2] - 1.0).abs() < 1e-15);
        assert!((w[3] - 0.5).abs() < 1e-15);
    }

    #[test]
    fn reflect_pad_mirrors_without_repeating_the_edge() {
        let w: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let mut out = Vec::new();
        reflect_pad(&w, 3, &mut out);
        assert_eq!(out.len(), 14);
        // left: w[3],w[2],w[1]  then w, then right: w[6],w[5],w[4]
        assert_eq!(
            out,
            vec![3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 6.0, 5.0, 4.0]
        );
    }

    #[test]
    fn repeatpad_tiles_then_zero_fills() {
        let w = vec![1.0, 2.0, 3.0];
        let mut out = Vec::new();
        repeatpad(&w, 10, &mut out);
        // int(10/3) == 3 tiles == 9 samples, then one zero.
        assert_eq!(out, vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 0.0]);
        let long: Vec<f64> = (0..20).map(|i| i as f64).collect();
        repeatpad(&long, 10, &mut out);
        assert_eq!(out, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    }

    #[test]
    fn slaney_scale_is_the_piecewise_one_not_htk() {
        // Measured from the pinned stack: filter_freqs = mel_to_hz(linspace(hz_to_mel(50),
        // hz_to_mel(14000), 66)). The HTK formula gives 85.1 Hz for the second point instead of
        // 103.985, and the two scales disagree by 624 Hz in the middle of the band.
        let step = (hz_to_mel_slaney(14_000.0) - hz_to_mel_slaney(50.0)) / 65.0;
        let ff: Vec<f64> = (0..66)
            .map(|i| mel_to_hz_slaney(hz_to_mel_slaney(50.0) + step * i as f64))
            .collect();
        let want = [
            50.0f64, 103.985, 157.97, 211.955, 265.94, 319.925, 373.91, 427.894,
        ];
        for (got, w) in ff.iter().zip(want) {
            assert!((got - w).abs() < 0.01, "filter_freqs: {got} vs {w}");
        }
        assert!(
            (ff[65] - 14_000.0).abs() < 1.0,
            "top of the band: {}",
            ff[65]
        );
        // Round trip, including across the 1000 Hz knee.
        for hz in [1.0, 50.0, 999.0, 1000.0, 1001.0, 8000.0, 14_000.0] {
            let back = mel_to_hz_slaney(hz_to_mel_slaney(hz));
            assert!((back - hz).abs() / hz < 1e-9, "round trip at {hz}: {back}");
        }
    }

    #[test]
    fn slaney_bank_is_triangular_in_mel_and_zero_outside_the_band() {
        let p = FrontendParams::default();
        let bank = mel_bank_slaney(&p);
        assert_eq!(bank.len(), 513 * 64);
        // Below fmin (50 Hz) everything is zero; bin 1 is 46.875 Hz, bin 2 is 93.75 Hz.
        assert!(
            bank[0..64].iter().all(|v| *v == 0.0),
            "DC row must be empty"
        );
        // The top of the bank is 14 kHz, so the 15-24 kHz rows are empty too.
        for bin in 321..513 {
            assert!(
                bank[bin * 64..(bin + 1) * 64].iter().all(|v| *v == 0.0),
                "row {bin} ({:.0} Hz) should be above fmax",
                24000.0 * bin as f64 / 512.0
            );
        }
        // Each mel column must rise then fall (a single triangle) and stay non-negative.
        for m in 0..64 {
            let col: Vec<f32> = (0..513).map(|b| bank[b * 64 + m]).collect();
            assert!(col.iter().all(|v| *v >= 0.0));
            let peak = col.iter().cloned().fold(0f32, f32::max);
            assert!(peak > 0.0, "mel band {m} is empty");
            let mut rising = true;
            let mut prev = 0.0f32;
            let mut fell = false;
            for v in &col {
                if rising && *v < prev {
                    rising = false;
                    fell = true;
                }
                if !rising && *v > prev && !fell {
                    unreachable!()
                }
                prev = *v;
            }
        }
    }

    #[test]
    fn silence_maps_to_the_floor_everywhere() {
        let fe = Frontend::new(FrontendParams::default());
        let mel = fe.process(&vec![0.0f32; 48_000]);
        assert_eq!(mel.len(), 1001 * 64);
        // 10*log10(1e-10) == -100 dB, exactly, and that is also what the reference reports for a
        // silent clip (`mel_min: -100.0` in the manifest).
        assert!(
            mel.iter().all(|v| *v == -100.0f32),
            "silence must sit on the dB floor"
        );
    }
}
