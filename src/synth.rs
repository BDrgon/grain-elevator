//! Frozen one-parameter synth sweeps. Quality observations are separate from correctness.
use crate::{Encoder, Error};
use serde::Serialize;
#[derive(Serialize)]
pub struct Observation {
    pub knob: String,
    pub values: [f64; 3],
    pub distances: [f64; 3],
    pub adjacent_12_within_span: bool,
    pub adjacent_23_within_span: bool,
}
#[derive(Serialize)]
pub struct Report {
    pub identity: String,
    pub metric: String,
    pub tolerance: f64,
    pub cases: Vec<Observation>,
    pub passed: usize,
    pub total: usize,
    pub pass_fraction: f64,
}
pub fn signal(knob: &str, value: f64, rate: u32) -> Vec<f32> {
    let mut result = Vec::with_capacity(rate as usize);
    let mut low = 0.0;
    for i in 0..rate {
        let t = i as f64 / rate as f64;
        let hz = if knob == "pitch" { value } else { 220.0 };
        let phase = 2.0 * std::f64::consts::PI * hz * t;
        let fm = if knob == "fm_depth" {
            value * (2.0 * std::f64::consts::PI * 110.0 * t).sin()
        } else {
            0.0
        };
        let harmonics = if knob == "harmonic_mix" { value } else { 0.3 };
        let raw = (phase + fm).sin()
            + harmonics * (2.0 * phase).sin()
            + harmonics * 0.5 * (3.0 * phase).sin();
        let cutoff = if knob == "filter_cutoff" {
            value
        } else {
            8000.0
        };
        let alpha = 1.0 - (-2.0 * std::f64::consts::PI * cutoff / rate as f64).exp();
        low += alpha * (raw - low);
        let decay = if knob == "decay" { value } else { 0.6 };
        let env = (t / 0.01).min(1.0) * (-t / decay).exp() * (1.0 - t).min(0.01).max(0.0) / 0.01;
        result.push((0.3 * low * env) as f32);
    }
    result
}
pub fn measure(encoder: &Encoder) -> Result<Report, Error> {
    let rate = if encoder.info.frontend == "clap-v1" {
        48000
    } else {
        16000
    };
    let cases = [
        ("pitch", [110., 220., 440.]),
        ("pitch", [220., 330., 440.]),
        ("harmonic_mix", [0., 0.3, 0.9]),
        ("harmonic_mix", [0.1, 0.5, 1.]),
        ("filter_cutoff", [300., 1000., 5000.]),
        ("filter_cutoff", [500., 2000., 8000.]),
        ("decay", [0.05, 0.2, 0.8]),
        ("decay", [0.1, 0.4, 1.]),
        ("fm_depth", [0., 1., 4.]),
        ("fm_depth", [0.5, 2., 6.]),
    ];
    let mut observations = vec![];
    let tolerance = 1e-6;
    for (knob, values) in cases {
        let a = encoder.embed_pcm(&signal(knob, values[0], rate), rate)?;
        let b = encoder.embed_pcm(&signal(knob, values[1], rate), rate)?;
        let c = encoder.embed_pcm(&signal(knob, values[2], rate), rate)?;
        let distances = [
            a.cosine_distance(&b)?,
            b.cosine_distance(&c)?,
            a.cosine_distance(&c)?,
        ];
        observations.push(Observation {
            knob: knob.into(),
            values,
            distances,
            adjacent_12_within_span: distances[0] <= distances[2] + tolerance,
            adjacent_23_within_span: distances[1] <= distances[2] + tolerance,
        });
    }
    let passed = observations
        .iter()
        .filter(|v| v.adjacent_12_within_span && v.adjacent_23_within_span)
        .count();
    let total = observations.len();
    Ok(Report {
        identity: encoder.info.identity(),
        metric: "cosine distance; raw tensors flattened with equal shapes/masks".into(),
        tolerance,
        cases: observations,
        passed,
        total,
        pass_fraction: passed as f64 / total as f64,
    })
}
