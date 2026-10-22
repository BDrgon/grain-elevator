//! Grain Elevator: decoded audio or mono PCM -> model-specific tensor/vector.
//! No indexing, playback, UI, model download, or Python runtime.
pub mod decode;
pub mod dsp;
pub mod error;
mod ffi;
pub mod fft;
pub mod gemma;
pub mod payload;
pub mod resample;
pub mod synth;

pub use error::Error;
use ort::{session::Session, value::Tensor};
pub use payload::ModelInfo;
use std::{
    borrow::Cow,
    path::{Path, PathBuf},
    sync::Mutex,
};

#[derive(Clone, Debug)]
pub struct Embedding {
    pub values: Vec<f32>,
    pub shape: Vec<usize>,
    pub mask: Vec<bool>,
    pub identity: String,
}
impl Embedding {
    pub fn cosine_distance(&self, other: &Self) -> Result<f64, Error> {
        if self.identity != other.identity
            || self.shape != other.shape
            || self.mask != other.mask
            || self.values.len() != other.values.len()
        {
            return Err(Error::artifact(
                "distance requires the same model, tensor shape and mask",
            ));
        }
        let (mut dot, mut a, mut b) = (0.0, 0.0, 0.0);
        for (&x, &y) in self.values.iter().zip(&other.values) {
            dot += x as f64 * y as f64;
            a += (x as f64).powi(2);
            b += (y as f64).powi(2);
        }
        if !a.is_finite() || !b.is_finite() || !dot.is_finite() || a <= 0.0 || b <= 0.0 {
            return Err(Error::Backend(
                "cosine requires finite, nonzero tensors".into(),
            ));
        }
        Ok((1.0 - dot / (a * b).sqrt()).clamp(0.0, 2.0))
    }
}
pub struct Encoder {
    pub info: ModelInfo,
    session: Mutex<Session>,
    clap: dsp::Frontend,
}
fn runtime_library() -> Result<PathBuf, Error> {
    if let Some(path) = std::env::var_os("GRAIN_ORT_LIB") {
        return Ok(path.into());
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    let exe = std::env::current_exe().map_err(|e| Error::io("locating executable", e))?;
    Ok(exe
        .parent()
        .ok_or_else(|| Error::artifact("no executable directory"))?
        .join(name))
}
fn init_runtime() -> Result<(), Error> {
    static INIT: std::sync::OnceLock<Result<(), String>> = std::sync::OnceLock::new();
    INIT.get_or_init(|| {
        let path = runtime_library().map_err(|e| e.to_string())?;
        ort::init_from(&path)
            .map(|b| {
                b.with_telemetry(false).commit();
            })
            .map_err(|e| format!("loading {}: {e}", path.display()))
    })
    .clone()
    .map_err(Error::Backend)
}
fn backend(e: impl std::fmt::Display) -> Error {
    Error::Backend(e.to_string())
}
impl Encoder {
    pub fn open() -> Result<Self, Error> {
        Self::open_image(
            std::env::current_exe().map_err(|e| Error::io("locating executable", e))?,
            None,
        )
    }
    /// Open weights embedded in a packaged executable or shared library.
    pub fn open_image(path: impl AsRef<Path>, threads: Option<usize>) -> Result<Self, Error> {
        let payload = payload::Payload::open(path.as_ref())?;
        init_runtime()?;
        let cap = threads.unwrap_or_else(|| {
            (std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
                / 2)
            .clamp(1, 4)
        });
        if cap == 0 {
            return Err(Error::Backend("threads must be at least one".into()));
        }
        let mut builder = Session::builder()
            .map_err(backend)?
            .with_intra_threads(cap)
            .map_err(backend)?
            .with_inter_threads(1)
            .map_err(backend)?;
        let graph = &payload.info.parts[0];
        for part in payload.info.parts.iter().skip(1) {
            builder = builder
                .with_external_initializer_file_in_memory(
                    &part.name,
                    Cow::Owned(payload.bytes(part).to_vec()),
                )
                .map_err(backend)?;
        }
        let session = builder
            .commit_from_memory(payload.bytes(graph))
            .map_err(backend)?;
        let expected: &[&str] = match (payload.info.frontend.as_str(), payload.info.output.as_str())
        {
            ("clap-v1", "embedding") => &["input_features"],
            ("gemma4-v1", "embedding") => &[
                "input_ids",
                "attention_mask",
                "input_features",
                "input_features_mask",
            ],
            ("gemma4-v1", "raw-tower") => &["input_features", "input_features_mask"],
            _ => return Err(Error::artifact("unsupported frontend/output pair")),
        };
        let names: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if names != expected {
            return Err(Error::artifact(format!(
                "graph inputs {:?}, expected {:?}",
                names, expected
            )));
        }
        let out: Vec<&str> = session.outputs().iter().map(|o| o.name()).collect();
        let expected_out: &[&str] = if payload.info.output == "raw-tower" {
            &["audio_features", "audio_mask"]
        } else {
            &["embedding"]
        };
        if out != expected_out {
            return Err(Error::artifact("graph outputs do not match catalog"));
        }
        Ok(Self {
            info: payload.info,
            session: Mutex::new(session),
            clap: dsp::Frontend::new(Default::default()),
        })
    }
    pub fn embed_file(&self, path: impl AsRef<Path>) -> Result<Embedding, Error> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| Error::io(format!("reading {}", path.display()), e))?;
        self.embed_bytes(&bytes, path.extension().and_then(|s| s.to_str()))
    }
    pub fn embed_bytes(&self, bytes: &[u8], extension: Option<&str>) -> Result<Embedding, Error> {
        let d = decode::decode_bytes_limit(
            bytes,
            extension,
            if self.info.frontend == "clap-v1" {
                10
            } else {
                30
            },
        )?;
        self.embed_pcm(&d.samples, d.sample_rate)
    }
    /// Mono PCM. CLAP uses the first 10 s; Gemma uses the first 30 s.
    pub fn embed_pcm(&self, samples: &[f32], rate: u32) -> Result<Embedding, Error> {
        if samples.is_empty() {
            return Err(Error::Empty);
        }
        if !(1000..=384000).contains(&rate) || samples.iter().any(|v| !v.is_finite()) {
            return Err(Error::Decode(
                "invalid sample rate or non-finite PCM".into(),
            ));
        }
        let target = if self.info.frontend == "clap-v1" {
            48000
        } else {
            16000
        };
        let seconds = if target == 48000 { 10 } else { 30 };
        let head = &samples[..samples.len().min(rate as usize * seconds)];
        let pcm = if rate == target {
            head.to_vec()
        } else {
            resample::resample_to(head, rate, target)
        };
        let mut session = self
            .session
            .lock()
            .map_err(|_| backend("encoder lock poisoned"))?;
        let outputs = if target == 48000 {
            let mel = self.clap.process(&pcm);
            let tensor = Tensor::from_array(([1usize, 1, 1001, 64], mel)).map_err(backend)?;
            session
                .run(ort::inputs!["input_features" => tensor])
                .map_err(backend)?
        } else {
            let f = gemma::features(&pcm)?;
            let features =
                Tensor::from_array(([1usize, f.frames, 128], f.values)).map_err(backend)?;
            let mask = Tensor::from_array(([1usize, f.frames], f.mask)).map_err(backend)?;
            if self.info.output == "raw-tower" {
                session
                    .run(ort::inputs!["input_features" => features, "input_features_mask" => mask])
                    .map_err(backend)?
            } else {
                let ids = gemma::input_ids(f.tokens);
                let count = ids.len();
                let input_ids = Tensor::from_array(([1usize, count], ids)).map_err(backend)?;
                let attention =
                    Tensor::from_array(([1usize, count], vec![1i64; count])).map_err(backend)?;
                session.run(ort::inputs!["input_ids" => input_ids,"attention_mask" => attention,"input_features" => features,"input_features_mask" => mask]).map_err(backend)?
            }
        };
        let (shape, data) = outputs[0].try_extract_tensor::<f32>().map_err(backend)?;
        let values = data.to_vec();
        if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
            return Err(backend("invalid output values"));
        }
        let shape: Vec<usize> = shape.iter().map(|&n| n as usize).collect();
        if self.info.output == "embedding"
            && (shape != [1, self.info.dimension] || values.len() != self.info.dimension)
        {
            return Err(Error::Dim {
                model: self.info.dimension,
                given: values.len(),
            });
        }
        let mask = if self.info.output == "raw-tower" {
            if shape.len() != 3 || shape[0] != 1 || shape[2] != self.info.dimension {
                return Err(backend("invalid tower output shape"));
            }
            let (_, mask) = outputs[1].try_extract_tensor::<bool>().map_err(backend)?;
            if mask.len() != shape[1] {
                return Err(backend("invalid tower mask shape"));
            }
            mask.to_vec()
        } else {
            vec![]
        };
        Ok(Embedding {
            values,
            shape,
            mask,
            identity: self.info.identity(),
        })
    }
}
