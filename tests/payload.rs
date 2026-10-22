use grain_elevator::payload::Payload;
use sha2::{Digest, Sha256};
#[test]
fn embedded_payload_rejects_corruption_and_bad_bounds() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(".build");
    std::fs::create_dir_all(&root).unwrap();
    let path = root.join(format!("payload-test-{}.bin", std::process::id()));
    let mut metadata = serde_json::json!({"schema":1,"key":"test","repo":"owner/repo","revision":"0000000000000000000000000000000000000000","artifact":"model.onnx","frontend":"clap-v1","output":"embedding","dimension":512,"quantization":"fp32","parts":[{"name":"model.onnx","offset":4,"length":5,"sha256":format!("{:x}",Sha256::digest(b"graph"))}]});
    let write = |metadata: &serde_json::Value, corrupt: bool| {
        let mut bytes = b"codegraph".to_vec();
        if corrupt {
            bytes[4] = 0;
        }
        let encoded = serde_json::to_vec(metadata).unwrap();
        bytes.extend(&encoded);
        bytes.extend((encoded.len() as u64).to_le_bytes());
        bytes.extend(b"GRAIN-ELEVATOR01");
        std::fs::write(&path, bytes).unwrap();
    };
    write(&metadata, false);
    assert!(Payload::open(&path).is_ok());
    write(&metadata, true);
    assert!(Payload::open(&path).is_err());
    metadata["parts"][0]["length"] = serde_json::json!(u64::MAX);
    write(&metadata, false);
    assert!(Payload::open(&path).is_err());
    std::fs::write(&path, b"short").unwrap();
    assert!(Payload::open(&path).is_err());
    std::fs::remove_file(path).unwrap();
}
