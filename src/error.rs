//! Typed errors for the embedding side (SPEC §3.1: a small, explicit set).

pub type Result<T> = std::result::Result<T, crate::error::Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("io: {context}: {source}")]
    Io {
        context: String,
        source: std::io::Error,
    },

    #[error("model artifact is invalid: {0}")]
    InvalidArtifact(String),

    /// A graph/frontend mismatch would produce plausible but incorrect vectors.
    #[error("frontend mismatch: model wants {want:?}, this build is configured {got:?}")]
    FrontendMismatch { want: String, got: String },

    #[error("unsupported audio format: {0}")]
    UnsupportedFormat(String),

    #[error("no decodable audio track")]
    NoAudioTrack,

    #[error("decode produced no audio samples")]
    Empty,

    #[error("decode failed: {0}")]
    Decode(String),

    #[error("inference failed: {0}")]
    Backend(String),

    /// A vector that is not the model's dimension can only come from a bug or a mismatched store;
    /// either way it must not become a silent zero.
    #[error("vector dimension mismatch: model emits {model}, got {given}")]
    Dim { model: usize, given: usize },

    #[error("cancelled")]
    Cancelled,
}

impl Error {
    pub fn io(context: impl Into<String>, source: std::io::Error) -> Error {
        Error::Io {
            context: context.into(),
            source,
        }
    }

    pub fn artifact(msg: impl Into<String>) -> Error {
        Error::InvalidArtifact(msg.into())
    }

    /// Alias used by the decoder for "the file is not what it claims".
    pub fn corrupt(msg: impl Into<String>) -> Error {
        Error::InvalidArtifact(msg.into())
    }
}
