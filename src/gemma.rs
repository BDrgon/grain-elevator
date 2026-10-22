//! Native implementation of the pinned Gemma4AudioFeatureExtractor audio path.
use crate::error::{Error, Result};
use crate::fft::Fft;

pub struct Features {
    pub values: Vec<f32>,
    pub mask: Vec<bool>,
    pub frames: usize,
    pub tokens: usize,
}

pub fn features(samples: &[f32]) -> Result<Features> {
    if samples.len() <= 160 {
        return Err(Error::Decode("Gemma needs more than 10 ms of audio".into()));
    }
    let n = samples.len().min(480_000);
    let padded = n.div_ceil(128) * 128;
    let frames = (padded + 160 - 321) / 160 + 1;
    let fft = Fft::new(512);
    let window: Vec<f32> = (0..320)
        .map(|i| (0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / 320.0).cos()) as f32)
        .collect();
    let mel_max = 2595.0 * (1.0f64 + 8000.0 / 700.0).log10();
    let bounds: Vec<f64> = (0..130)
        .map(|i| 700.0 * (10f64.powf(mel_max * i as f64 / 129.0 / 2595.0) - 1.0))
        .collect();
    let mut bank = vec![0f64; 257 * 128];
    for k in 0..257 {
        let hz = k as f64 * 8000.0 / 256.0;
        for m in 0..128 {
            bank[k * 128 + m] = ((hz - bounds[m]) / (bounds[m + 1] - bounds[m]))
                .min((bounds[m + 2] - hz) / (bounds[m + 2] - bounds[m + 1]))
                .max(0.0);
        }
    }
    let mut values = Vec::with_capacity(frames * 128);
    let mut mask = Vec::with_capacity(frames);
    for frame in 0..frames {
        let valid = frame * 160 + 320 < n + 160;
        mask.push(valid);
        let mut re = vec![0f64; 512];
        let mut im = vec![0f64; 512];
        for i in 0..320 {
            let pos = frame * 160 + i;
            let value = if pos >= 160 && pos - 160 < n {
                samples[pos - 160]
            } else {
                0.0
            };
            // NumPy multiplies the float32 waveform/window before its float64 FFT.
            re[i] = (value * window[i]) as f64;
        }
        fft.forward(&mut re, &mut im);
        let magnitudes: Vec<f64> = (0..257)
            // The pinned NumPy stack rounds FFT output to complex64, then
            // computes float32 magnitudes before its float64 mel multiply.
            .map(|k| (re[k] as f32).hypot(im[k] as f32) as f64)
            .collect();
        for m in 0..128 {
            let mut sum = 0.0;
            for k in 0..257 {
                sum += magnitudes[k] * bank[k * 128 + m];
            }
            values.push(if valid {
                (sum + 0.001).ln() as f32
            } else {
                0.0
            });
        }
    }
    let mut token_mask = mask.clone();
    for _ in 0..2 {
        let count = token_mask.len().div_ceil(2);
        token_mask = token_mask.into_iter().step_by(2).take(count).collect();
    }
    let tokens = token_mask.iter().filter(|&&v| v).count();
    if tokens == 0 {
        return Err(Error::Decode("Gemma produced no valid audio tokens".into()));
    }
    Ok(Features {
        values,
        mask,
        frames,
        tokens,
    })
}

pub fn input_ids(tokens: usize) -> Vec<i64> {
    let mut ids = vec![2, 256000];
    ids.extend(std::iter::repeat_n(258881, tokens));
    ids.extend([258883, 1]);
    ids
}
