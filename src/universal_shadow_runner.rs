use crate::error::{BrainError, BrainResult};
use crate::materialization_selector::ComparativeControl;
use crate::shadow_evaluation::{ShadowEvaluationBundle, ShadowRuntimeOutput};
use serde_json::Value;
use std::collections::BTreeSet;
use std::io::{Read, Write};

const RUNNER_INPUT_PATH: &str = "/tidex/input";
const MAX_INPUT_BYTES: u64 = 256 * 1024 * 1024;
const MIN_SCHEMA: &str = "cerebro.tidex.shadow_evaluation_bundle/v1";
const MAX_EXPLICIT_CONTROLS: usize = 64;

fn parse_probability(value: Option<&Value>, field: &str) -> BrainResult<f64> {
    let value = value
        .ok_or_else(|| BrainError::Invalid(format!("shadow_runner_metric_missing:{field}")))?;
    let number = value
        .as_f64()
        .ok_or_else(|| BrainError::Invalid(format!("shadow_runner_metric_invalid:{field}")))?;
    if number.is_finite() && (0.0..=1.0).contains(&number) {
        Ok(number)
    } else {
        Err(BrainError::Invalid(format!(
            "shadow_runner_metric_out_of_range:{field}"
        )))
    }
}

fn parse_u64_value(value: &Value) -> Option<u64> {
    if let Some(value) = value.as_u64() {
        return Some(value);
    }
    let value = value.as_f64()?;
    if !value.is_finite() || value < 0.0 || value.fract() != 0.0 {
        return None;
    }
    if value > u64::MAX as f64 {
        return None;
    }
    Some(value as u64)
}

fn parse_u64_signal(value: Option<&Value>, field: &str) -> BrainResult<u64> {
    let value =
        value.ok_or_else(|| BrainError::Invalid(format!("shadow_runner_u64_missing:{field}")))?;
    let parsed = parse_u64_value(value)
        .ok_or_else(|| BrainError::Invalid(format!("shadow_runner_u64_invalid:{field}")))?;
    if parsed == 0 {
        return Err(BrainError::Invalid(format!(
            "shadow_runner_u64_out_of_range:{field}"
        )));
    }
    Ok(parsed)
}

fn parse_control_name(value: &str) -> Option<ComparativeControl> {
    match value {
        "unmodified_receiver" => Some(ComparativeControl::UnmodifiedReceiver),
        "wrong_capability_ir" => Some(ComparativeControl::WrongCapabilityIr),
        "random_delta" => Some(ComparativeControl::RandomDelta),
        "mean_capability" => Some(ComparativeControl::MeanCapability),
        "nearest_capability" => Some(ComparativeControl::NearestCapability),
        "alternative_backend" => Some(ComparativeControl::AlternativeBackend),
        "non_target_preservation" => Some(ComparativeControl::NonTargetPreservation),
        "dense_delta" => Some(ComparativeControl::DenseDelta),
        "conventional_low_rank" => Some(ComparativeControl::ConventionalLowRank),
        "sparse_delta" => Some(ComparativeControl::SparseDelta),
        "activation_steering" => Some(ComparativeControl::ActivationSteering),
        _ => None,
    }
}

fn parse_comparative_controls(
    value: Option<&Value>,
    required: bool,
) -> BrainResult<BTreeSet<ComparativeControl>> {
    let Some(value) = value else {
        if required {
            return Err(BrainError::Invalid("shadow_runner_controls_missing".into()));
        }
        return Ok(BTreeSet::new());
    };
    let Value::Array(values) = value else {
        return Err(BrainError::Invalid("shadow_runner_controls_invalid".into()));
    };
    if values.is_empty() || values.len() > MAX_EXPLICIT_CONTROLS {
        return Err(BrainError::Invalid(
            "shadow_runner_controls_cardinality_invalid".into(),
        ));
    }
    let mut controls = BTreeSet::new();
    for raw in values {
        let name = raw
            .as_str()
            .ok_or_else(|| BrainError::Invalid("shadow_runner_control_invalid".into()))?;
        let control = parse_control_name(name)
            .ok_or_else(|| BrainError::Invalid("shadow_runner_control_unknown".into()))?;
        controls.insert(control);
    }
    if controls.len() != values.len() {
        return Err(BrainError::Invalid(
            "shadow_runner_control_duplicate".into(),
        ));
    }
    Ok(controls)
}

fn controls_from_payload(
    payload: &Value,
    required: bool,
) -> BrainResult<BTreeSet<ComparativeControl>> {
    let mut controls = BTreeSet::new();

    if let Some(metrics) = payload.get("metrics") {
        controls.extend(parse_comparative_controls(
            metrics
                .get("completed_controls")
                .or_else(|| metrics.get("controls")),
            false,
        )?);
    }

    controls.extend(parse_comparative_controls(
        payload
            .get("completed_controls")
            .or_else(|| payload.get("controls")),
        required && controls.is_empty(),
    )?);
    controls.extend(parse_comparative_controls(
        payload
            .get("evidence")
            .and_then(|value| value.get("comparative_controls")),
        false,
    )?);
    Ok(controls)
}

#[derive(Debug, Clone)]
struct EvidenceSignals {
    functional_score: Option<f64>,
    functional_ci_lower: Option<f64>,
    preservation_score: Option<f64>,
    identity_margin: Option<f64>,
    numerical_stability: Option<f64>,
    normalized_risk: Option<f64>,
    latency_micros: Option<u64>,
    resident_bytes: Option<u64>,
    completed_controls: BTreeSet<ComparativeControl>,
}

impl EvidenceSignals {
    fn empty() -> Self {
        Self {
            functional_score: None,
            functional_ci_lower: None,
            preservation_score: None,
            identity_margin: None,
            numerical_stability: None,
            normalized_risk: None,
            latency_micros: None,
            resident_bytes: None,
            completed_controls: BTreeSet::new(),
        }
    }
}

#[derive(Debug, Clone)]
struct EvaluatedSignals {
    functional_score: f64,
    functional_ci_lower: f64,
    preservation_score: f64,
    identity_margin: f64,
    numerical_stability: f64,
    normalized_risk: f64,
    latency_micros: u64,
    resident_bytes: u64,
    completed_controls: BTreeSet<ComparativeControl>,
}

fn parse_payload_signal(payload: &[u8], require_metrics: bool) -> BrainResult<EvidenceSignals> {
    let parsed = match serde_json::from_slice::<Value>(payload) {
        Ok(payload) => payload,
        Err(_) if require_metrics => {
            return Err(BrainError::Invalid(
                "shadow_runner_payload_invalid_json".into(),
            ));
        }
        Err(_) => return Ok(EvidenceSignals::empty()),
    };

    if require_metrics {
        let root = parsed.get("metrics").unwrap_or(&parsed);
        let Some(root_object) = root.as_object() else {
            return Err(BrainError::Invalid(
                "shadow_runner_metric_payload_not_object".into(),
            ));
        };

        let functional_score =
            parse_probability(root_object.get("functional_score"), "functional_score")?;
        let functional_ci_lower = parse_probability(
            root_object.get("functional_ci_lower"),
            "functional_ci_lower",
        )?;
        if functional_ci_lower > functional_score {
            return Err(BrainError::Invalid(
                "shadow_runner_metric_invalid:functional_ci_lower".into(),
            ));
        }
        let preservation_score =
            parse_probability(root_object.get("preservation_score"), "preservation_score")?;
        let identity_margin =
            parse_probability(root_object.get("identity_margin"), "identity_margin")?;
        let numerical_stability = parse_probability(
            root_object.get("numerical_stability"),
            "numerical_stability",
        )?;
        let normalized_risk =
            parse_probability(root_object.get("normalized_risk"), "normalized_risk")?;
        let latency_micros = parse_u64_signal(
            root.get("latency_micros")
                .or_else(|| root.get("latency_us"))
                .or_else(|| root.get("latency")),
            "latency_micros",
        )?;
        let resident_bytes = parse_u64_signal(
            root.get("resident_bytes")
                .or_else(|| root.get("resident_bytes_estimate")),
            "resident_bytes",
        )?;

        Ok(EvidenceSignals {
            functional_score: Some(functional_score),
            functional_ci_lower: Some(functional_ci_lower),
            preservation_score: Some(preservation_score),
            identity_margin: Some(identity_margin),
            numerical_stability: Some(numerical_stability),
            normalized_risk: Some(normalized_risk),
            latency_micros: Some(latency_micros),
            resident_bytes: Some(resident_bytes),
            completed_controls: controls_from_payload(root, true)?,
        })
    } else {
        Ok(EvidenceSignals {
            completed_controls: controls_from_payload(&parsed, false)?,
            ..EvidenceSignals::empty()
        })
    }
}

fn merge_controls(
    evaluation: &EvidenceSignals,
    candidate: &EvidenceSignals,
    receiver: &EvidenceSignals,
) -> BTreeSet<ComparativeControl> {
    let mut controls = BTreeSet::new();
    controls.extend(evaluation.completed_controls.iter().cloned());
    controls.extend(candidate.completed_controls.iter().cloned());
    controls.extend(receiver.completed_controls.iter().cloned());
    controls
}

fn evaluate_signal(
    evaluation: &EvidenceSignals,
    candidate: &EvidenceSignals,
    receiver: &EvidenceSignals,
) -> BrainResult<EvaluatedSignals> {
    let completed_controls = merge_controls(evaluation, candidate, receiver);
    let (
        Some(functional_score),
        Some(functional_ci_lower),
        Some(preservation_score),
        Some(identity_margin),
        Some(numerical_stability),
        Some(normalized_risk),
        Some(latency_micros),
        Some(resident_bytes),
    ) = (
        evaluation.functional_score,
        evaluation.functional_ci_lower,
        evaluation.preservation_score,
        evaluation.identity_margin,
        evaluation.numerical_stability,
        evaluation.normalized_risk,
        evaluation.latency_micros,
        evaluation.resident_bytes,
    )
    else {
        return Err(BrainError::Invalid("shadow_runner_metrics_missing".into()));
    };

    if completed_controls.is_empty() {
        return Err(BrainError::Invalid("shadow_runner_controls_missing".into()));
    }

    Ok(EvaluatedSignals {
        functional_score,
        functional_ci_lower,
        preservation_score,
        identity_margin,
        numerical_stability,
        normalized_risk,
        latency_micros,
        resident_bytes,
        completed_controls,
    })
}

fn metrics(bundle: &ShadowEvaluationBundle) -> BrainResult<ShadowRuntimeOutput> {
    let eval = parse_payload_signal(&bundle.evaluation_payload, true)?;
    let candidate = parse_payload_signal(&bundle.candidate_payload, false)?;
    let receiver = parse_payload_signal(&bundle.receiver_payload, false)?;
    let signals = evaluate_signal(&eval, &candidate, &receiver)?;
    Ok(ShadowRuntimeOutput {
        schema: "cerebro.tidex.shadow_runtime_output/v1".into(),
        receiver_snapshot_sha256: bundle.receiver_snapshot_sha256.clone(),
        candidate_sha256: bundle.candidate_sha256.clone(),
        receiver_payload_sha256: bundle.receiver_payload_sha256.clone(),
        candidate_payload_sha256: bundle.candidate_payload_sha256.clone(),
        evaluation_payload_sha256: bundle.evaluation_payload_sha256.clone(),
        functional_score: signals.functional_score,
        functional_ci_lower: signals.functional_ci_lower,
        preservation_score: signals.preservation_score,
        identity_margin: signals.identity_margin,
        numerical_stability: signals.numerical_stability,
        normalized_risk: signals.normalized_risk,
        latency_micros: signals.latency_micros,
        resident_bytes: signals.resident_bytes,
        completed_controls: signals.completed_controls,
        optimizer_steps: bundle.optimizer_steps,
    })
}

fn bundle_load(bytes: &[u8]) -> crate::error::BrainResult<ShadowEvaluationBundle> {
    let bundle: ShadowEvaluationBundle = serde_json::from_slice(bytes)?;
    bundle.validate()?;
    Ok(bundle)
}

pub fn run_universal_shadow_runner() -> crate::error::BrainResult<()> {
    let input_path = std::env::var_os("TIDEX_INPUT_PATH")
        .ok_or_else(|| BrainError::Invalid("tidex_input_path_missing".into()))?;
    if std::path::Path::new(&input_path) != std::path::Path::new(RUNNER_INPUT_PATH) {
        return Err(BrainError::Invalid("tidex_input_path_invalid".into()));
    }

    let file = std::fs::File::open(&input_path)?;
    let metadata = file.metadata()?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_INPUT_BYTES {
        return Err(BrainError::Invalid(
            "shadow_runner_input_file_invalid".into(),
        ));
    }

    let mut bytes = Vec::new();
    let read_limit = MAX_INPUT_BYTES
        .checked_add(1)
        .ok_or_else(|| BrainError::Invalid("shadow_runner_read_limit_overflow".into()))?;
    std::io::BufReader::new(file)
        .take(read_limit)
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())
        .map_err(|_| BrainError::Invalid("shadow_runner_package_length_overflow".into()))?
        > MAX_INPUT_BYTES
    {
        return Err(BrainError::Invalid("shadow_runner_bundle_too_large".into()));
    }
    if serde_json::from_slice::<serde_json::Value>(&bytes)
        .map_err(|_| BrainError::Invalid("shadow_runner_input_not_json".into()))?
        .get("schema")
        .and_then(|value| value.as_str())
        .is_none_or(|schema| schema != MIN_SCHEMA)
    {
        return Err(BrainError::Invalid(
            "shadow_runner_input_schema_invalid".into(),
        ));
    }

    let bundle = bundle_load(&bytes)?;
    let output = metrics(&bundle)?;
    let stdout = std::io::stdout();
    let mut lock = stdout.lock();
    let output_bytes = serde_json::to_vec_pretty(&output)?;
    lock.write_all(&output_bytes)?;
    lock.write_all(b"\n")?;
    lock.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::digest::Sha256Digest;
    use crate::receiver_profile::MaterializationStrategy;
    use crate::shadow_evaluation::ShadowEvaluationBundle;
    use std::collections::BTreeSet;

    fn evaluation_payload() -> Vec<u8> {
        r#"{"functional_score":0.93,"functional_ci_lower":0.90,"preservation_score":0.99,"identity_margin":0.84,"numerical_stability":0.998,"normalized_risk":0.01,"latency_micros":1200,"resident_bytes":2048,"completed_controls":["unmodified_receiver","dense_delta","wrong_capability_ir","random_delta","mean_capability","nearest_capability","alternative_backend","non_target_preservation"]}"#
            .as_bytes()
            .to_vec()
    }

    fn fixture() -> ShadowEvaluationBundle {
        ShadowEvaluationBundle::create(
            Sha256Digest::digest_bytes(b"receiver"),
            Sha256Digest::digest_bytes(b"candidate"),
            MaterializationStrategy::DenseDelta,
            b"receiver-payload".to_vec(),
            b"candidate-payload".to_vec(),
            evaluation_payload(),
        )
        .unwrap()
    }

    #[test]
    fn complete_controls_cover_required_set() {
        let bundle = fixture();
        let output = metrics(&bundle).unwrap();
        assert_eq!(output.schema, "cerebro.tidex.shadow_runtime_output/v1");
        assert!(output.functional_ci_lower <= output.functional_score);
        assert!(output.functional_ci_lower >= 0.0);
        assert!(output.numerical_stability <= 1.0);
        assert_eq!(output.optimizer_steps, 0);
        assert!(output
            .completed_controls
            .contains(&ComparativeControl::DenseDelta));
    }

    #[test]
    fn explicit_metrics_are_respected() {
        let bundle = fixture();
        let output = metrics(&bundle).unwrap();
        assert!((output.functional_score - 0.93).abs() < 1e-12);
        assert!((output.functional_ci_lower - 0.90).abs() < 1e-12);
        assert_eq!(output.latency_micros, 1200);
        assert_eq!(output.resident_bytes, 2048);
        assert!(output
            .completed_controls
            .contains(&ComparativeControl::DenseDelta));
        assert!(output.identity_margin >= 0.8);
    }

    #[test]
    fn metrics_are_bounded() {
        let mut bundle = fixture();
        bundle.strategy = MaterializationStrategy::SparseDelta;
        let output = metrics(&bundle).unwrap();
        let checks = [
            output.functional_score,
            output.functional_ci_lower,
            output.preservation_score,
            output.identity_margin,
            output.numerical_stability,
            output.normalized_risk,
        ];
        for value in checks {
            assert!(value.is_finite());
            assert!((0.0..=1.0).contains(&value));
        }
        assert!(output
            .completed_controls
            .contains(&ComparativeControl::DenseDelta));
        assert!(!output
            .completed_controls
            .contains(&ComparativeControl::SparseDelta));
        let _ =
            BTreeSet::<ComparativeControl>::from_iter(output.completed_controls.iter().cloned());
    }

    #[test]
    fn runner_does_not_invent_strategy_controls() {
        let mut bundle = fixture();
        bundle.strategy = MaterializationStrategy::ActivationSteering;
        bundle.evaluation_payload = r#"{"functional_score":0.91,"functional_ci_lower":0.89,"preservation_score":0.98,"identity_margin":0.7,"numerical_stability":0.997,"normalized_risk":0.02,"latency_micros":1200,"resident_bytes":2048,"completed_controls":["unmodified_receiver"]}"#
            .as_bytes()
            .to_vec();
        bundle.evaluation_payload_sha256 = Sha256Digest::digest_bytes(&bundle.evaluation_payload);
        let output = metrics(&bundle).unwrap();
        assert_eq!(
            output.completed_controls,
            BTreeSet::from([ComparativeControl::UnmodifiedReceiver])
        );
    }

    #[test]
    fn missing_metrics_must_fail() {
        let mut bundle = fixture();
        bundle.evaluation_payload = b"{\"invalid\":1}".to_vec();
        assert!(metrics(&bundle).is_err());
    }

    #[test]
    fn missing_controls_must_fail() {
        let mut bundle = fixture();
        bundle.evaluation_payload = r#"{"functional_score":0.93,"functional_ci_lower":0.90,"preservation_score":0.99,"identity_margin":0.84,"numerical_stability":0.998,"normalized_risk":0.01,"latency_micros":1200,"resident_bytes":2048}"#
            .as_bytes()
            .to_vec();
        bundle.evaluation_payload_sha256 = Sha256Digest::digest_bytes(&bundle.evaluation_payload);
        assert!(metrics(&bundle).is_err());
    }

    #[test]
    fn zero_latency_or_memory_must_fail() {
        let mut bundle = fixture();
        bundle.evaluation_payload = r#"{"functional_score":0.93,"functional_ci_lower":0.90,"preservation_score":0.99,"identity_margin":0.84,"numerical_stability":0.998,"normalized_risk":0.01,"latency_micros":0,"resident_bytes":2048,"completed_controls":["unmodified_receiver"]}"#
            .as_bytes()
            .to_vec();
        bundle.evaluation_payload_sha256 = Sha256Digest::digest_bytes(&bundle.evaluation_payload);
        assert!(metrics(&bundle).is_err());

        bundle.evaluation_payload = r#"{"functional_score":0.93,"functional_ci_lower":0.90,"preservation_score":0.99,"identity_margin":0.84,"numerical_stability":0.998,"normalized_risk":0.01,"latency_micros":1200,"resident_bytes":0,"completed_controls":["unmodified_receiver"]}"#
            .as_bytes()
            .to_vec();
        bundle.evaluation_payload_sha256 = Sha256Digest::digest_bytes(&bundle.evaluation_payload);
        assert!(metrics(&bundle).is_err());
    }

    #[test]
    fn receiver_and_candidate_payloads_use_controls_if_json() {
        let mut bundle = fixture();
        bundle.receiver_payload = b"{\"completed_controls\":[\"mean_capability\"]}".to_vec();
        bundle.candidate_payload = b"not-json".to_vec();
        let output = metrics(&bundle).unwrap();
        assert!(output
            .completed_controls
            .contains(&ComparativeControl::MeanCapability));
    }
}
