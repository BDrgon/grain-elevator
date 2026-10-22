use grain_elevator::{gemma, Embedding};
#[test]
fn gemma_matches_pinned_processor() {
    let data: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/gemma_frontend.json")).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let samples: Vec<f32> = serde_json::from_value(case["samples"].clone()).unwrap();
        let expected: Vec<f32> = serde_json::from_value(case["features"].clone()).unwrap();
        let mask: Vec<bool> = serde_json::from_value(case["mask"].clone()).unwrap();
        let ids: Vec<i64> = serde_json::from_value(case["ids"].clone()).unwrap();
        let actual = gemma::features(&samples).unwrap();
        assert_eq!(actual.mask, mask);
        assert_eq!(actual.values.len(), expected.len());
        assert_eq!(gemma::input_ids(actual.tokens), ids);
        let error = actual
            .values
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(error <= 1e-5, "Gemma feature error {error}");
    }
}
#[test]
fn rejects_invalid_distance_and_empty_audio() {
    assert!(gemma::features(&[]).is_err());
    assert!(gemma::features(&[0.; 160]).is_err());
    let a = Embedding {
        values: vec![1., 0.],
        shape: vec![1, 2],
        mask: vec![],
        identity: "a".into(),
    };
    let b = Embedding {
        values: vec![1., 0.],
        shape: vec![1, 2],
        mask: vec![],
        identity: "b".into(),
    };
    assert!(a.cosine_distance(&b).is_err());
    assert_eq!(a.cosine_distance(&a).unwrap(), 0.);
    let mut invalid = a.clone();
    invalid.values.push(1.);
    assert!(a.cosine_distance(&invalid).is_err());
    invalid = a.clone();
    invalid.values[0] = f32::NAN;
    assert!(a.cosine_distance(&invalid).is_err());
}
