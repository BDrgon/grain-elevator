//! Embedded executable/library overlay. Large weights bypass linker object-size limits.
use crate::error::{Error, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs::File, path::Path};

const MAGIC: &[u8; 16] = b"GRAIN-ELEVATOR01";
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Part {
    pub name: String,
    pub offset: u64,
    pub length: u64,
    pub sha256: String,
}
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ModelInfo {
    pub schema: u32,
    pub key: String,
    pub repo: String,
    pub revision: String,
    pub artifact: String,
    pub frontend: String,
    pub output: String,
    pub dimension: usize,
    pub quantization: String,
    pub parts: Vec<Part>,
}
impl ModelInfo {
    pub fn identity(&self) -> String {
        format!(
            "{}@{}/{}:{}:{}",
            self.repo, self.revision, self.artifact, self.frontend, self.output
        )
    }
}
pub struct Payload {
    pub info: ModelInfo,
    map: Mmap,
}
impl Payload {
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| Error::io("opening embedded image", e))?;
        // Immutable mapping; the image must not be modified during encoder lifetime.
        let map =
            unsafe { Mmap::map(&file) }.map_err(|e| Error::io("mapping embedded image", e))?;
        if map.len() < 24 || &map[map.len() - 16..] != MAGIC {
            return Err(Error::artifact(
                "image has no grain-elevator model payload; use tools/build.py",
            ));
        }
        let footer = map.len() - 24;
        let len = u64::from_le_bytes(map[footer..footer + 8].try_into().unwrap());
        let len = usize::try_from(len)
            .map_err(|_| Error::artifact("manifest length overflows platform"))?;
        if len > 1024 * 1024 || len > footer {
            return Err(Error::artifact("invalid payload manifest length"));
        }
        let start = footer - len;
        let info: ModelInfo = serde_json::from_slice(&map[start..footer])
            .map_err(|e| Error::artifact(format!("manifest: {e}")))?;
        if info.schema != 1
            || !["clap-v1", "gemma4-v1"].contains(&info.frontend.as_str())
            || !["embedding", "raw-tower"].contains(&info.output.as_str())
            || info.parts.is_empty()
        {
            return Err(Error::artifact("unsupported model contract"));
        }
        for part in &info.parts {
            let end = part
                .offset
                .checked_add(part.length)
                .ok_or_else(|| Error::artifact("payload offset overflow"))?;
            if end > start as u64 {
                return Err(Error::artifact("payload part outside image"));
            }
            let bytes = &map[part.offset as usize..end as usize];
            let digest = format!("{:x}", Sha256::digest(bytes));
            if digest != part.sha256 {
                return Err(Error::artifact(format!(
                    "payload checksum failed: {}",
                    part.name
                )));
            }
        }
        Ok(Self { info, map })
    }
    pub fn bytes(&self, part: &Part) -> &[u8] {
        &self.map[part.offset as usize..(part.offset + part.length) as usize]
    }
}
