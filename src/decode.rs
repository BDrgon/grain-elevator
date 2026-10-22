//! Decode: file bytes → mono float32 at 48 kHz, the way the reference does it.
//!
//! The reference reads with libsndfile (and ffmpeg for the containers libsndfile does not open),
//! averages to mono, keeps the **first `min(len, sample_rate*10)` frames at the source rate**, and
//! only then resamples (engine.py:40). Order matters: capping after resampling would keep a
//! different number of samples for a 44.1 kHz file, and the mel would be shifted by a fraction of
//! a frame.
//!
//! symphonia 0.6 covers the whole SPEC §5 format list in pure Rust — wav/aiff (riff), mp3, flac,
//! ogg/vorbis, aac + alac in isomp4/m4a, caf, adpcm, raw pcm — so there is no ffmpeg anywhere in
//! the runtime dependency tree. ffmpeg is used by `tools/` to *make* fixtures, nothing more.

use std::io::Cursor;

use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::errors::Error as SymError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;

use crate::error::{Error, Result};
use crate::resample;

/// The reference's window: 10 seconds, taken at the source rate before resampling.
pub const REFERENCE_SECONDS: u32 = 10;
pub const TARGET_RATE: u32 = 48_000;

/// What a decode produced, before the reference's 10 s cap and resampling.
#[derive(Clone, Debug)]
pub struct Decoded {
    /// Mono, float32, at `sample_rate`.
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    /// Channels found in the file (1 for mono, 2 for stereo, …) — reported for the file-info
    /// table, not used after the downmix.
    pub channels: usize,
    /// Frames decoded (per channel, after the downmix).
    pub frames: usize,
    /// False when the decode stopped early because it already had 10 s of audio.
    pub decoded_all: bool,
}

impl Decoded {
    pub fn duration_s(&self) -> f32 {
        if self.sample_rate == 0 {
            0.0
        } else {
            self.frames as f32 / self.sample_rate as f32
        }
    }
}

/// Decode `path` and return exactly what the frontend wants: mono float32 at 48 kHz, at most
/// 10 s, capped at the source rate first.
pub fn decode_for_embed(path: &std::path::Path) -> Result<Vec<f32>> {
    let bytes =
        std::fs::read(path).map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
    let hint = path.extension().and_then(|e| e.to_str());
    let d = decode_bytes(&bytes, hint)?;
    Ok(prepare(&d))
}

/// The reference's preparation, split out so it can be tested against the `signal/*.f32` fixtures
/// without a file in sight.
pub fn prepare(d: &Decoded) -> Vec<f32> {
    // First min(frames, src_rate * 10) frames, at the source rate.
    let cap = (d.sample_rate * REFERENCE_SECONDS) as usize;
    let n = d.samples.len().min(cap);
    let head = &d.samples[..n];
    if d.sample_rate == TARGET_RATE {
        return head.to_vec();
    }
    resample::resample_to(head, d.sample_rate, TARGET_RATE)
}

/// Decode bytes into mono float32 at whatever rate the file declares.
///
/// `extension` is only a probe hint (`.m4a` vs `.aac` genuinely need it: one is ISO MP4, the other
/// ADTS). Content sniffing happens first.
pub fn decode_bytes(bytes: &[u8], extension: Option<&str>) -> Result<Decoded> {
    decode_bytes_limit(bytes, extension, REFERENCE_SECONDS)
}

pub fn decode_bytes_limit(bytes: &[u8], extension: Option<&str>, seconds: u32) -> Result<Decoded> {
    if bytes.is_empty() {
        return Err(Error::Empty);
    }
    let mut hint = Hint::new();
    if let Some(ext) = extension {
        hint.with_extension(ext);
    }
    let mss = MediaSourceStream::new(Box::new(Cursor::new(bytes)), Default::default());
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| match e {
            SymError::Unsupported(_) => Error::UnsupportedFormat(describe(&e)),
            other => Error::Decode(describe(&other)),
        })?;

    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| Error::NoAudioTrack)?;
    let track_id = track.id;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|p| p.audio())
        .ok_or_else(|| Error::corrupt("audio track has no codec parameters"))?;
    let declared_rate = params.sample_rate.unwrap_or(0);
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(|e| Error::UnsupportedFormat(describe(&e)))?;

    // Some streams do not declare a rate until the first frame decodes.
    let mut rate: u32 = declared_rate;

    let mut mono: Vec<f32> = Vec::new();
    let mut channels = 1usize;
    let mut decoded_all = true;

    loop {
        let packet = match format.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break,
            // A mid-stream track change (chained Ogg); the reference reads the first stream only.
            Err(SymError::ResetRequired) => break,
            Err(e) => {
                // A truncated tail is normal for lossy files with a sloppy writer. If we already
                // have audio, keep it; if we have nothing, that is a real failure.
                if mono.is_empty() {
                    return Err(Error::Decode(describe(&e)));
                }
                decoded_all = false;
                break;
            }
        };
        if packet.track_id != track_id {
            continue;
        }
        let buf = match decoder.decode(&packet) {
            Ok(b) => b,
            // Damaged packets inside an otherwise readable file: skip them, like the example does.
            Err(SymError::DecodeError(_)) | Err(SymError::IoError(_)) => continue,
            Err(e) => {
                if mono.is_empty() {
                    return Err(Error::Decode(describe(&e)));
                }
                decoded_all = false;
                break;
            }
        };
        if rate == 0 {
            rate = buf.spec().rate();
        }
        channels = buf.spec().channels().count().max(1);
        let mut interleaved: Vec<f32> = Vec::new();
        buf.copy_to_vec_interleaved::<f32>(&mut interleaved);
        downmix(&interleaved, channels, &mut mono);
        // 10 seconds is all the frontend can use, so stop rather than decode a feature-length
        // file for the first ten seconds of it.
        if mono.len() >= (rate as usize * seconds as usize + 8) {
            decoded_all = false;
            break;
        }
    }

    if mono.is_empty() {
        return Err(Error::Empty);
    }
    let frames = mono.len();
    Ok(Decoded {
        samples: mono,
        sample_rate: rate,
        channels,
        frames,
        decoded_all,
    })
}

/// Average `channels` interleaved planes into mono. numpy does `x.mean(axis=1)` on the decoded
/// array, which is the same arithmetic; doing it in float32 keeps us within a rounding step.
fn downmix(interleaved: &[f32], channels: usize, out: &mut Vec<f32>) {
    if channels <= 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    out.reserve(interleaved.len() / channels);
    for frame in interleaved.chunks_exact(channels) {
        let mut acc = 0.0f32;
        for s in frame {
            acc += *s;
        }
        out.push(acc / channels as f32);
    }
}

fn describe(e: &SymError) -> String {
    match e {
        SymError::Unsupported(reason) => format!("unsupported: {reason}"),
        SymError::DecodeError(msg) => format!("decode error: {msg}"),
        SymError::IoError(e) => format!("io: {e}"),
        other => format!("{other:?}"),
    }
}
