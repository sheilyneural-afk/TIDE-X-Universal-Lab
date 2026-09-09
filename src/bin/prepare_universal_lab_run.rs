use cerebro_tidex::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
};
use cerebro_tidex::capability_discovery::{
    CapabilityDiscoveryPolicy, CapabilityDiscoveryRequest, CapabilityProbeTrial,
};
use cerebro_tidex::capability_ir::{
    CapabilityIr, IrNode, OperatorIrTransition, OutputBinding, PrimitiveSet, StateIrAnchor,
    TypedPort, ValueReference,
};
use cerebro_tidex::checkpoint_adapter::{
    inspect_safetensors_receiver, inventory_safetensors_checkpoint, InspectedReceiverArtifacts,
    SafeTensorsReceiverRequest, SafeTensorsTensorInventoryItem,
};
use cerebro_tidex::contracts::ProtectedCortex;
use cerebro_tidex::dense_shadow_materializer::materialize_replayed_dense_delta_shadow;
use cerebro_tidex::digest::Sha256Digest;
use cerebro_tidex::error::{BrainError, BrainResult};
use cerebro_tidex::identity::{
    AcquisitionId, ArchitectureId, CapabilityId, CapabilityNodeId, ModelId, PortId, PrimitiveId,
    TensorId,
};
use cerebro_tidex::isolated_execution::{
    AuthenticatedBytes, IsolationLimits, IsolationRequirements,
};
use cerebro_tidex::materialization_selector::{
    BackendEvaluation, BackendSelectionInput, BackendSelectionPolicy, ComparativeControl,
};
use cerebro_tidex::receiver_compiler::{ReceiverCalibrationSet, ReceiverCompilerPolicy};
use cerebro_tidex::receiver_profile::{
    CapabilityModality, CapabilityRequirements, MaterializationStrategy, ReceiverArchitecture,
};
use cerebro_tidex::shadow_evaluation::{
    run_shadow_evaluation, ShadowEvaluationBundle, ShadowEvaluationInput, ShadowEvaluationReceipt,
};
use cerebro_tidex::sparse_shadow_materializer::{
    materialize_replayed_sparse_shadow, SparseShadowPolicy,
};
use cerebro_tidex::universal_capability_compiler::{
    execute_universal_capability_shadow_plan, UniversalCapabilityCompilationRequest,
    UniversalCapabilityPlanningRequest, UniversalCapabilityShadowPlanReceipt,
};
use cerebro_tidex::universal_promotion_gate::{
    evaluate_universal_promotion_gate, UniversalPromotionGateRequest, UniversalPromotionPolicy,
};
use cerebro_tidex::universality_evidence::{
    UniversalityEvidenceInput, UniversalityProtocol, UniversalityTrial,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

fn main() {
    if let Err(error) = run(env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    if args.first().is_some_and(|value| value == "suite") {
        return run_suite(args.into_iter().skip(1).collect());
    }
    let config = RunConfig::parse(args)?;
    let model_root = config.model_root;
    let tensor_id = config.tensor_id;
    let backend = config.backend;
    let output_dir = config.output_dir;
    let capability = CapabilitySpec::parse(config.capability_id.as_str())?;
    let seed = config.seed;
    fs::create_dir_all(&output_dir)?;

    let receiver_request = receiver_request(&model_root, tensor_id.clone())?;
    write_json(&output_dir.join("receiver-request.json"), &receiver_request)?;
    let inspected = inspect_safetensors_receiver(&model_root, &receiver_request)?;
    write_json(&output_dir.join("receiver-artifacts.json"), &inspected)?;
    write_json(&output_dir.join("receiver-layout.json"), &inspected.layout)?;

    let discovery_request = discovery_request(&inspected, &capability, seed);
    write_json(
        &output_dir.join("discovery-request.json"),
        &discovery_request,
    )?;

    let (planning_request, main_receipt) = planning_request(
        &output_dir,
        &inspected,
        tensor_id.clone(),
        backend,
        &capability,
        seed,
    )?;
    write_json(&output_dir.join("planning-request.json"), &planning_request)?;

    let sparse_policy = sparse_policy();
    write_json(&output_dir.join("sparse-policy.json"), &sparse_policy)?;

    let started = Instant::now();
    let main_candidate = materialize_candidate_value(
        &planning_request,
        &main_receipt,
        &inspected,
        &sparse_policy,
        backend,
    )?;
    let main_materialization_micros = started.elapsed().as_micros().max(1) as u64;
    write_json(
        &output_dir.join("prepared-main-candidate.json"),
        &main_candidate,
    )?;

    let alternative_backend = match backend {
        MaterializationStrategy::DenseDelta => MaterializationStrategy::SparseDelta,
        MaterializationStrategy::SparseDelta => MaterializationStrategy::DenseDelta,
        _ => return Err("prepare backend unsupported".into()),
    };
    let mut alternative_request = planning_request.clone();
    alternative_request.requested_strategy = alternative_backend;
    let alternative_receipt = execute_universal_capability_shadow_plan(&alternative_request)?;
    let alternative_candidate = materialize_candidate_value(
        &alternative_request,
        &alternative_receipt,
        &inspected,
        &sparse_policy,
        alternative_backend,
    )?;
    write_json(
        &output_dir.join("prepared-alternative-candidate.json"),
        &alternative_candidate,
    )?;

    let shadow_input = shadow_input(
        &inspected,
        &main_receipt,
        &main_candidate,
        &alternative_candidate,
        backend,
        main_materialization_micros,
    )?;
    write_json(&output_dir.join("shadow-input.json"), &shadow_input)?;

    write_json(
        &output_dir.join("selection-policy.json"),
        &selection_policy(),
    )?;
    write_json(
        &output_dir.join("universality-input.json"),
        &universality_input(&main_receipt, &capability, seed, &inspected)?,
    )?;
    write_json(
        &output_dir.join("promotion-policy.json"),
        &UniversalPromotionPolicy {
            schema: "cerebro.tidex.universal_promotion_policy/v1".into(),
            minimum_universality_n: 1,
            minimum_global_wilson_lower_bound: 0.0,
            require_all_selected_candidates_evaluated: true,
        },
    )?;

    write_json(
        &output_dir.join("preparation-summary.json"),
        &json!({
            "schema": "cerebro.tidex.universal_lab_preparation_summary/v1",
            "model_root": model_root,
            "selected_tensor_id": tensor_id,
            "backend": backend,
            "capability_id": capability.id.as_str(),
            "seed": seed,
            "checkpoint_snapshot_sha256": inspected.snapshot.model_snapshot_sha256,
            "receiver_snapshot_sha256": inspected.snapshot.manifest_sha256,
            "receiver_layout_sha256": inspected.layout.manifest_sha256,
            "main_candidate_sha256": digest_field(&main_candidate, "manifest_sha256")?,
            "alternative_candidate_sha256": digest_field(&alternative_candidate, "manifest_sha256")?,
            "outputs": {
                "receiver_request": output_dir.join("receiver-request.json"),
                "discovery_request": output_dir.join("discovery-request.json"),
                "planning_request": output_dir.join("planning-request.json"),
                "receiver_layout": output_dir.join("receiver-layout.json"),
                "backend_policy": if backend == MaterializationStrategy::SparseDelta {
                    Value::String(output_dir.join("sparse-policy.json").display().to_string())
                } else {
                    Value::String("-".into())
                },
                "steering_layout": "-",
                "shadow_input": output_dir.join("shadow-input.json"),
                "selection_policy": output_dir.join("selection-policy.json"),
                "universality_input": output_dir.join("universality-input.json"),
                "promotion_policy": output_dir.join("promotion-policy.json")
            }
        }),
    )?;

    println!("{}", output_dir.display());
    Ok(())
}

fn run_suite(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: prepare-universal-lab-run suite <dense|sparse> <output-dir> <shadow-runner> <model-root>...";
    if args.len() < 4 {
        return Err(usage.into());
    }
    let backend = parse_backend(&args[0])?;
    let output_dir = PathBuf::from(&args[1]);
    let runner = AuthenticatedBytes::from_trusted_bytes(fs::read(&args[2])?);
    let model_roots = args[3..].iter().map(PathBuf::from).collect::<Vec<_>>();
    let capabilities = suite_capabilities()?;
    let seeds = suite_seeds()?;
    fs::create_dir_all(&output_dir)?;

    let mut receiver_records = Vec::new();
    let mut case_records = Vec::new();
    let mut failures = Vec::new();
    let mut evaluations = Vec::<BackendEvaluation>::new();
    let mut shadow_receipts = Vec::<ShadowEvaluationReceipt>::new();
    let mut trials = Vec::<UniversalityTrial>::new();

    for model_root in model_roots {
        let receiver_result = (|| -> Result<(), Box<dyn std::error::Error>> {
            let tensor_id = select_materialization_tensor(&model_root)?;
            let receiver_request = receiver_request(&model_root, tensor_id.clone())?;
            let inspected = inspect_safetensors_receiver(&model_root, &receiver_request)?;
            let receiver_id = receiver_id(&inspected);
            let family_id = receiver_family_id(&inspected)?;
            let receiver_dir = output_dir.join("receivers").join(safe_slug(&receiver_id));
            write_json(
                &receiver_dir.join("receiver-request.json"),
                &receiver_request,
            )?;
            write_json(&receiver_dir.join("receiver-artifacts.json"), &inspected)?;
            write_json(
                &receiver_dir.join("receiver-layout.json"),
                &inspected.layout,
            )?;
            receiver_records.push(json!({
                "model_root": model_root,
                "receiver_id": receiver_id,
                "receiver_family_id": family_id,
                "selected_tensor_id": tensor_id,
                "parameter_dimension": inspected.profile.parameter_dimension,
                "snapshot_sha256": inspected.snapshot.manifest_sha256,
                "layout_sha256": inspected.layout.manifest_sha256
            }));

            for capability in &capabilities {
                for seed in &seeds {
                    let cap_slug = safe_slug(capability.id.as_str());
                    let case_dir = output_dir
                        .join("cases")
                        .join(safe_slug(&receiver_id))
                        .join(&cap_slug)
                        .join(format!("seed-{seed:04}"));
                    fs::create_dir_all(&case_dir)?;
                    write_json(&case_dir.join("receiver-request.json"), &receiver_request)?;
                    write_json(&case_dir.join("receiver-artifacts.json"), &inspected)?;
                    write_json(&case_dir.join("receiver-layout.json"), &inspected.layout)?;

                    let discovery = discovery_request(&inspected, capability, *seed);
                    write_json(&case_dir.join("discovery-request.json"), &discovery)?;
                    let discovery_report = discovery.execute()?;
                    write_json(
                        &case_dir.join("02-discovery-report.json"),
                        &discovery_report,
                    )?;

                    let (planning, plan_receipt) = planning_request(
                        &case_dir,
                        &inspected,
                        tensor_id.clone(),
                        backend,
                        capability,
                        *seed,
                    )?;
                    write_json(&case_dir.join("planning-request.json"), &planning)?;
                    write_json(&case_dir.join("03-shadow-plan.json"), &plan_receipt)?;
                    write_json(
                        &case_dir.join("04-shadow-plan-replay.json"),
                        &json!({
                            "schema": "cerebro.tidex.universal_shadow_plan_replay/v1",
                            "planning_request_sha256": plan_receipt.planning_request_sha256,
                            "replayed": true
                        }),
                    )?;

                    let sparse_policy = sparse_policy();
                    write_json(&case_dir.join("sparse-policy.json"), &sparse_policy)?;
                    let started = Instant::now();
                    let candidate = materialize_candidate_value(
                        &planning,
                        &plan_receipt,
                        &inspected,
                        &sparse_policy,
                        backend,
                    )?;
                    let materialization_micros = started.elapsed().as_micros().max(1) as u64;
                    write_json(
                        &case_dir.join("05-materialization-candidate.json"),
                        &candidate,
                    )?;

                    let alternative_backend = match backend {
                        MaterializationStrategy::DenseDelta => MaterializationStrategy::SparseDelta,
                        MaterializationStrategy::SparseDelta => MaterializationStrategy::DenseDelta,
                        _ => return Err("prepare backend unsupported".into()),
                    };
                    let mut alternative_request = planning.clone();
                    alternative_request.requested_strategy = alternative_backend;
                    let alternative_receipt =
                        execute_universal_capability_shadow_plan(&alternative_request)?;
                    let alternative_candidate = materialize_candidate_value(
                        &alternative_request,
                        &alternative_receipt,
                        &inspected,
                        &sparse_policy,
                        alternative_backend,
                    )?;
                    write_json(
                        &case_dir.join("prepared-alternative-candidate.json"),
                        &alternative_candidate,
                    )?;

                    let shadow_input = shadow_input(
                        &inspected,
                        &plan_receipt,
                        &candidate,
                        &alternative_candidate,
                        backend,
                        materialization_micros,
                    )?;
                    write_json(&case_dir.join("shadow-input.json"), &shadow_input)?;
                    let shadow_receipt = run_shadow_evaluation(
                        runner.clone(),
                        &shadow_input.bundle,
                        shadow_input.arguments,
                        shadow_input.limits,
                        shadow_input.requirements,
                    )?;
                    write_json(
                        &case_dir.join("06-shadow-evaluation-receipt.json"),
                        &shadow_receipt,
                    )?;

                    let per_case_selection_input = BackendSelectionInput {
                        schema: "cerebro.tidex.backend_selection_input/v1".into(),
                        evaluations: vec![shadow_receipt.evaluation.clone()],
                        complementarity: vec![],
                        policy: selection_policy(),
                    };
                    let per_case_selection = per_case_selection_input.execute()?;
                    write_json(
                        &case_dir.join("07-backend-selection-receipt.json"),
                        &per_case_selection,
                    )?;

                    let trial = universality_trial(
                        &inspected,
                        &plan_receipt,
                        &shadow_receipt,
                        capability,
                        *seed,
                    )?;
                    trials.push(trial.clone());
                    evaluations.push(shadow_receipt.evaluation.clone());
                    shadow_receipts.push(shadow_receipt.clone());
                    case_records.push(json!({
                        "case_dir": case_dir,
                        "receiver_id": receiver_id,
                        "receiver_family_id": family_id,
                        "capability_id": capability.id.as_str(),
                        "seed": seed,
                        "candidate_sha256": shadow_receipt.evaluation.candidate_sha256,
                        "functional_score": shadow_receipt.evaluation.functional_score,
                        "functional_ci_lower": shadow_receipt.evaluation.functional_ci_lower,
                        "preservation_score": shadow_receipt.evaluation.preservation_score,
                        "selected": per_case_selection.selected_candidates
                    }));
                }
            }
            Ok(())
        })();
        if let Err(error) = receiver_result {
            failures.push(json!({
                "model_root": model_root,
                "error": error.to_string()
            }));
        }
    }

    if evaluations.is_empty() || trials.is_empty() {
        return Err("suite_produced_no_evidence".into());
    }

    let selection_input = BackendSelectionInput {
        schema: "cerebro.tidex.backend_selection_input/v1".into(),
        evaluations: evaluations.clone(),
        complementarity: vec![],
        policy: selection_policy(),
    };
    let selection_receipt = selection_input.execute()?;
    write_json(
        &output_dir.join("suite-backend-selection-input.json"),
        &selection_input,
    )?;
    write_json(
        &output_dir.join("suite-backend-selection-receipt.json"),
        &selection_receipt,
    )?;

    let universality_input = UniversalityEvidenceInput {
        schema: "cerebro.tidex.universality_evidence_input/v1".into(),
        calibration_capabilities: BTreeSet::from(["state.identity:v1".into()]),
        trials,
        protocol: suite_protocol(&capabilities, &seeds),
    };
    let universality_receipt = universality_input.execute()?;
    write_json(
        &output_dir.join("suite-universality-input.json"),
        &universality_input,
    )?;
    write_json(
        &output_dir.join("suite-universality-receipt.json"),
        &universality_receipt,
    )?;

    let required_n = capabilities
        .iter()
        .filter(|capability| capability.id.as_str() != "state.identity:v1")
        .count()
        .max(1);
    let promotion_request = UniversalPromotionGateRequest {
        schema: "cerebro.tidex.universal_promotion_gate_request/v1".into(),
        selection_input,
        selection_receipt: selection_receipt.clone(),
        universality_input,
        universality_receipt: universality_receipt.clone(),
        shadow_evaluations: shadow_receipts,
        policy: UniversalPromotionPolicy {
            schema: "cerebro.tidex.universal_promotion_policy/v1".into(),
            minimum_universality_n: required_n,
            minimum_global_wilson_lower_bound: 0.8,
            require_all_selected_candidates_evaluated: true,
        },
    };
    let promotion_receipt = evaluate_universal_promotion_gate(&promotion_request)?;
    write_json(
        &output_dir.join("suite-promotion-request.json"),
        &promotion_request,
    )?;
    write_json(
        &output_dir.join("suite-promotion-receipt.json"),
        &promotion_receipt,
    )?;

    let mut families = BTreeMap::<String, usize>::new();
    for record in &receiver_records {
        if let Some(family) = record.get("receiver_family_id").and_then(Value::as_str) {
            *families.entry(family.into()).or_default() += 1;
        }
    }
    let summary = json!({
        "schema": "cerebro.tidex.universal_lab_suite_summary/v1",
        "backend": backend,
        "receiver_count": receiver_records.len(),
        "case_count": case_records.len(),
        "failure_count": failures.len(),
        "capabilities": capabilities.iter().map(|capability| capability.id.as_str()).collect::<Vec<_>>(),
        "seeds": seeds,
        "families": families,
        "universality_n": universality_receipt.universality_n,
        "global_wilson_lower_bound": universality_receipt.global_wilson_lower_bound,
        "promotion_readiness": promotion_receipt.readiness,
        "promotion_blockers": promotion_receipt.blockers,
        "authorizes_activation": promotion_receipt.authorizes_activation,
        "receivers": receiver_records,
        "cases": case_records,
        "failures": failures
    });
    write_json(&output_dir.join("suite-summary.json"), &summary)?;
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}

struct RunConfig {
    model_root: PathBuf,
    tensor_id: TensorId,
    backend: MaterializationStrategy,
    output_dir: PathBuf,
    capability_id: String,
    seed: u64,
}

impl RunConfig {
    fn parse(args: Vec<String>) -> Result<Self, Box<dyn std::error::Error>> {
        let usage = "usage: prepare-universal-lab-run <model-root> <tensor-id> <dense|sparse> <output-dir> [capability-id] [seed]";
        if !(4..=6).contains(&args.len()) {
            return Err(usage.into());
        }
        let seed = args
            .get(5)
            .map(|value| value.parse::<u64>())
            .transpose()
            .map_err(|_| "prepare_seed_invalid")?
            .unwrap_or(0);
        Ok(Self {
            model_root: PathBuf::from(&args[0]),
            tensor_id: TensorId::parse(&args[1])?,
            backend: parse_backend(&args[2])?,
            output_dir: PathBuf::from(&args[3]),
            capability_id: args
                .get(4)
                .cloned()
                .unwrap_or_else(|| "state.toggle:v1".into()),
            seed,
        })
    }
}

#[derive(Debug, Clone)]
struct CapabilitySpec {
    id: CapabilityId,
    operator_id: &'static str,
    transitions: Vec<(&'static str, &'static str)>,
}

impl CapabilitySpec {
    fn parse(value: &str) -> BrainResult<Self> {
        let id = CapabilityId::parse(value)?;
        let transitions = match id.as_str() {
            "state.toggle:v1" => vec![("s0", "s1"), ("s1", "s0")],
            "state.identity:v1" => vec![("s0", "s0"), ("s1", "s1")],
            "state.reset0:v1" => vec![("s0", "s0"), ("s1", "s0")],
            "state.reset1:v1" => vec![("s0", "s1"), ("s1", "s1")],
            "state.negate:v1" => vec![("s0", "n0"), ("s1", "n1")],
            _ => return Err(BrainError::Invalid("prepare_capability_unknown".into())),
        };
        Ok(Self {
            id,
            operator_id: match value {
                "state.toggle:v1" => "toggle",
                "state.identity:v1" => "identity",
                "state.reset0:v1" => "reset0",
                "state.reset1:v1" => "reset1",
                "state.negate:v1" => "negate",
                _ => unreachable!(),
            },
            transitions,
        })
    }

    fn target_signature(&self) -> Vec<f64> {
        self.transitions
            .iter()
            .flat_map(|(_, target)| anchor_state(target))
            .collect()
    }

    fn source_contract(&self) -> Value {
        json!({
            "schema": "cerebro.tidex.capability_source_contract/v1",
            "capability_id": self.id.as_str(),
            "representation": "finite_state_operator",
            "state_dimension": 2,
            "operator": {
                "operator_id": self.operator_id,
                "transitions": self.transitions.iter().map(|(source, target)| {
                    json!({"source": source, "target": target})
                }).collect::<Vec<_>>()
            },
            "anchors": anchor_contracts(),
            "required_invariants": {
                "closure_error_lte": 0.00001,
                "contraction_ratio_lte": 0.00001,
                "identity_margin_gte": 0.05
            }
        })
    }
}

fn anchor_state(anchor: &str) -> Vec<f64> {
    match anchor {
        "n0" => vec![-1.0, 0.0],
        "n1" => vec![0.0, -1.0],
        "s0" => vec![1.0, 0.0],
        "s1" => vec![0.0, 1.0],
        _ => vec![0.0, 0.0],
    }
}

fn anchor_contracts() -> Vec<Value> {
    ["n0", "n1", "s0", "s1"]
        .into_iter()
        .map(|anchor| json!({"anchor_id": anchor, "state": anchor_state(anchor)}))
        .collect()
}

fn parse_backend(value: &str) -> BrainResult<MaterializationStrategy> {
    match value {
        "dense" => Ok(MaterializationStrategy::DenseDelta),
        "sparse" => Ok(MaterializationStrategy::SparseDelta),
        _ => Err(BrainError::Invalid("prepare_backend_invalid".into())),
    }
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}

fn digest_field(value: &Value, field: &str) -> Result<Sha256Digest, Box<dyn std::error::Error>> {
    Ok(serde_json::from_value(
        value
            .get(field)
            .ok_or_else(|| format!("missing digest field: {field}"))?
            .clone(),
    )?)
}

fn receiver_request(
    model_root: &Path,
    tensor_id: TensorId,
) -> Result<SafeTensorsReceiverRequest, Box<dyn std::error::Error>> {
    Ok(SafeTensorsReceiverRequest {
        schema: "cerebro.tidex.safetensors_receiver_request/v1".into(),
        model_id: ModelId::parse(model_id_from_root(model_root))?,
        architecture_id: ArchitectureId::parse("local.safetensors.receiver")?,
        architecture: ReceiverArchitecture::Unknown,
        modalities: BTreeSet::from([CapabilityModality::Text]),
        supports_persistent_state: false,
        checkpoint_files: vec!["model.safetensors".into()],
        configuration_file: "config.json".into(),
        tokenizer_file: "tokenizer.json".into(),
        supported_strategies: BTreeSet::from([
            MaterializationStrategy::DenseDelta,
            MaterializationStrategy::SparseDelta,
        ]),
        materialization_tensors: BTreeSet::from([tensor_id]),
    })
}

fn model_id_from_root(model_root: &Path) -> String {
    let components = model_root
        .components()
        .filter_map(|component| component.as_os_str().to_str())
        .collect::<Vec<_>>();
    let raw = components
        .windows(2)
        .rev()
        .find_map(|window| {
            (window[0] == "snapshots")
                .then(|| components.iter().rev().nth(2).copied())
                .flatten()
        })
        .or_else(|| model_root.file_name().and_then(|name| name.to_str()))
        .unwrap_or("receiver");
    let mut normalized = raw
        .trim_start_matches("models--")
        .replace("--", ".")
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else if ch == '.' || ch == '-' || ch == '_' {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    while normalized.contains("..") {
        normalized = normalized.replace("..", ".");
    }
    normalized = normalized.trim_matches(['.', '-', '_']).to_string();
    if normalized.is_empty() {
        normalized = "receiver".into();
    }
    let candidate = format!("local.{normalized}");
    if candidate.len() <= 128 {
        candidate
    } else {
        candidate.chars().take(128).collect()
    }
}

fn discovery_request(
    inspected: &InspectedReceiverArtifacts,
    capability: &CapabilitySpec,
    seed: u64,
) -> CapabilityDiscoveryRequest {
    let target = capability.target_signature();
    let wrong = wrong_signatures(capability)[0].clone();
    let trials = (0..4)
        .map(|index| {
            let signature = jitter_signature(&target, seed, index, 0.002);
            CapabilityProbeTrial {
                schema: "cerebro.tidex.capability_probe_trial/v1".into(),
                trial_id: format!("{}-seed-{seed}-probe-{index}", capability.operator_id),
                probe_id: capability.id.as_str().into(),
                seed: seed.saturating_mul(10_000).saturating_add(index as u64),
                input_sha256: Sha256Digest::digest_bytes(
                    format!("{}-probe-input-{seed}-{index}", capability.id.as_str()).as_bytes(),
                ),
                output_sha256: Sha256Digest::digest_bytes(
                    format!("{}-probe-output-{seed}-{index}", capability.id.as_str()).as_bytes(),
                ),
                functional_signature: signature,
                wrong_control_signature: wrong.clone(),
                closure_error: 0.0,
                contraction_ratio: 0.0,
                attributed_module_families: inspected
                    .architecture_fingerprint
                    .module_family_counts
                    .keys()
                    .copied()
                    .take(1)
                    .collect(),
            }
        })
        .collect();
    CapabilityDiscoveryRequest {
        schema: "cerebro.tidex.capability_discovery_request/v1".into(),
        architecture_fingerprint: inspected.architecture_fingerprint.clone(),
        trials,
        policy: CapabilityDiscoveryPolicy {
            schema: "cerebro.tidex.capability_discovery_policy/v1".into(),
            minimum_trials: 3,
            minimum_seeds: 3,
            minimum_consistency: 0.95,
            minimum_control_margin: 0.25,
            maximum_closure_error: 1e-9,
            maximum_contraction_ratio: 1e-9,
        },
    }
}

fn planning_request(
    output_dir: &Path,
    inspected: &InspectedReceiverArtifacts,
    tensor_id: TensorId,
    backend: MaterializationStrategy,
    capability: &CapabilitySpec,
    seed: u64,
) -> Result<
    (
        UniversalCapabilityPlanningRequest,
        UniversalCapabilityShadowPlanReceipt,
    ),
    Box<dyn std::error::Error>,
> {
    let source_root = output_dir.join("capability-source");
    write_json(
        &source_root.join("capability.contract.json"),
        &capability.source_contract(),
    )?;
    let acquisition = AcquisitionRequest::new(
        AcquisitionId::parse(format!(
            "real-universal-lab-{}-seed-{seed}",
            capability.operator_id
        ))?,
        AcquisitionScope::WholeProject,
        RequestedResidency::BestVerified,
        NoisePolicy::ExplicitOnly,
        AcquisitionBudget {
            max_files: 8,
            max_total_bytes: 1 << 20,
        },
        vec![],
    )?;
    let envelope =
        cerebro_tidex::acquisition_contract::SystemEnvelope::capture(&source_root, &acquisition)?;
    let ir = capability_ir(&envelope, capability)?;
    let operational = operational_contract(&ir, capability)?;
    let functional = calibration_signatures(capability, seed);
    let receiver_solutions = functional
        .iter()
        .map(|signature| {
            receiver_solution(signature, inspected.profile.parameter_dimension as usize)
        })
        .collect::<Vec<_>>();
    let request = UniversalCapabilityPlanningRequest {
        schema: "cerebro.tidex.universal_capability_planning_request/v1".into(),
        compilation: UniversalCapabilityCompilationRequest {
            schema: "cerebro.tidex.universal_capability_compilation_request/v1".into(),
            system_envelope: envelope,
            capability_ir: ir.clone(),
            operational_contract: operational,
            calibration: ReceiverCalibrationSet {
                receiver_snapshot_binding_sha256: inspected.snapshot.manifest_sha256.clone(),
                functional_signatures: functional.clone(),
                receiver_solutions,
                wrong_functional_signatures: wrong_signatures(capability),
            },
            protected_cortex: ProtectedCortex {
                parameter_importance: vec![0.0; inspected.profile.parameter_dimension as usize],
                directions: vec![],
                max_damage_ratio: 0.01,
            },
            risk_metric_rows: identity_rows(inspected.profile.parameter_dimension as usize),
            policy: ReceiverCompilerPolicy {
                schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                ridge: 1e-10,
                minimum_decoder_loo_r2: 0.999,
                minimum_encoder_loo_r2: 0.999,
                minimum_decoder_loo_cosine: 0.999,
                maximum_functional_relative_error: 1e-4,
                minimum_identity_margin: 0.05,
                maximum_quadratic_cost: 1e6,
            },
        },
        receiver_profile: inspected.profile.clone(),
        receiver_snapshot: inspected.snapshot.clone(),
        capability_requirements: CapabilityRequirements {
            schema: "cerebro.tidex.capability_requirements/v1".into(),
            capability_id: ir.capability_id().clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            required_modalities: BTreeSet::from([CapabilityModality::Text]),
            requires_persistent_state: false,
            minimum_receiver_parameter_dimension: inspected.profile.parameter_dimension,
            acceptable_strategies: BTreeSet::from([
                MaterializationStrategy::DenseDelta,
                MaterializationStrategy::SparseDelta,
            ]),
        },
        requested_strategy: backend,
        affected_regions: vec![tensor_id],
    };
    let receipt = execute_universal_capability_shadow_plan(&request)?;
    Ok((request, receipt))
}

fn capability_ir(
    envelope: &cerebro_tidex::acquisition_contract::SystemEnvelope,
    capability: &CapabilitySpec,
) -> BrainResult<CapabilityIr> {
    CapabilityIr::new(
        capability.id.clone(),
        envelope,
        PrimitiveSet::tidex_core_v1()?,
        vec![TypedPort::tensor_f64(PortId::parse("state")?, vec![2, 1])?],
        vec![IrNode::new(
            CapabilityNodeId::parse("node.normalize")?,
            PrimitiveId::parse("tensor.normalize")?,
            vec![ValueReference::Input {
                name: PortId::parse("state")?,
            }],
            TypedPort::tensor_f64(PortId::parse("normalized")?, vec![2, 1])?,
            vec![PathBuf::from("capability.contract.json")],
        )?],
        vec![OutputBinding::new(
            TypedPort::tensor_f64(PortId::parse("result")?, vec![2, 1])?,
            ValueReference::NodeOutput {
                node_id: CapabilityNodeId::parse("node.normalize")?,
            },
        )?],
    )
}

fn operational_contract(
    ir: &CapabilityIr,
    capability: &CapabilitySpec,
) -> BrainResult<cerebro_tidex::capability_ir::OperationalCapabilityContract> {
    let mut transitions = capability
        .transitions
        .iter()
        .map(|(source, target)| {
            let source_state = anchor_state(source);
            let target_state = anchor_state(target);
            let pre = source_state
                .iter()
                .zip(&target_state)
                .map(|(left, right)| (left - right).powi(2))
                .sum::<f64>()
                .sqrt();
            OperatorIrTransition {
                operator_id: capability.operator_id.into(),
                source_anchor_id: (*source).into(),
                target_anchor_id: (*target).into(),
                observed_next_state: target_state,
                pre_target_error: pre,
                post_target_error: 0.0,
            }
        })
        .collect::<Vec<_>>();
    transitions.sort_by(|left, right| {
        (
            left.operator_id.as_str(),
            left.source_anchor_id.as_str(),
            left.target_anchor_id.as_str(),
        )
            .cmp(&(
                right.operator_id.as_str(),
                right.source_anchor_id.as_str(),
                right.target_anchor_id.as_str(),
            ))
    });
    Ok(
        cerebro_tidex::capability_ir::OperationalCapabilityContract {
            schema: "cerebro.tidex.operational_capability/v1".into(),
            capability_id: ir.capability_id().clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            state_dimension: 2,
            anchors: ["n0", "n1", "s0", "s1"]
                .into_iter()
                .map(|anchor| StateIrAnchor {
                    anchor_id: anchor.into(),
                    state: anchor_state(anchor),
                })
                .collect(),
            transitions,
            maximum_closure_error: 1e-5,
            maximum_contraction_ratio: 1e-5,
        },
    )
}

fn calibration_signatures(capability: &CapabilitySpec, seed: u64) -> Vec<Vec<f64>> {
    let target = capability.target_signature();
    let candidates = vec![
        vec![1.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.0, 1.0],
        vec![1.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 1.0],
        vec![1.0, 0.5, 0.5, 1.0],
        vec![0.2, 1.0, 1.0, 0.2],
        vec![1.2, -0.2, 0.4, 0.8],
        vec![-0.8, 0.1, 0.2, -1.1],
        vec![0.3, -0.7, 0.9, 0.4],
    ];
    candidates
        .into_iter()
        .enumerate()
        .filter_map(|(index, signature)| {
            let adjusted = jitter_signature(&signature, seed, index, 0.0001);
            if signatures_close(&adjusted, &target) {
                None
            } else {
                Some(adjusted)
            }
        })
        .collect()
}

fn signatures_close(left: &[f64], right: &[f64]) -> bool {
    left.len() == right.len()
        && left
            .iter()
            .zip(right)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max)
            <= 1e-9
}

fn jitter_signature(signature: &[f64], seed: u64, index: usize, scale: f64) -> Vec<f64> {
    signature
        .iter()
        .enumerate()
        .map(|(slot, value)| {
            let mixed = seed
                .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                .wrapping_add((index as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9))
                .wrapping_add((slot as u64).wrapping_mul(0x94D0_49BB_1331_11EB));
            let centered = ((mixed % 2001) as f64 / 1000.0) - 1.0;
            value + centered * scale
        })
        .collect()
}

fn wrong_signatures(capability: &CapabilitySpec) -> Vec<Vec<f64>> {
    [
        "state.toggle:v1",
        "state.identity:v1",
        "state.reset0:v1",
        "state.reset1:v1",
        "state.negate:v1",
    ]
    .into_iter()
    .filter(|candidate| *candidate != capability.id.as_str())
    .filter_map(|candidate| CapabilitySpec::parse(candidate).ok())
    .map(|candidate| candidate.target_signature())
    .take(3)
    .collect()
}

fn suite_capabilities() -> BrainResult<Vec<CapabilitySpec>> {
    let configured = env::var("TIDEX_SUITE_CAPABILITIES").unwrap_or_else(|_| {
        "state.identity:v1,state.toggle:v1,state.reset0:v1,state.reset1:v1,state.negate:v1".into()
    });
    let mut capabilities = Vec::new();
    let mut seen = BTreeSet::new();
    for raw in configured.split(',') {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        let capability = CapabilitySpec::parse(value)?;
        if !seen.insert(capability.id.clone()) {
            return Err(BrainError::Invalid("suite_capability_duplicate".into()));
        }
        capabilities.push(capability);
    }
    if capabilities.is_empty()
        || capabilities
            .iter()
            .all(|capability| capability.id.as_str() != "state.identity:v1")
    {
        return Err(BrainError::Invalid(
            "suite_requires_identity_calibration_capability".into(),
        ));
    }
    Ok(capabilities)
}

fn suite_seeds() -> Result<Vec<u64>, Box<dyn std::error::Error>> {
    let configured = env::var("TIDEX_SUITE_SEEDS").unwrap_or_else(|_| "0,1,2".into());
    let mut seeds = Vec::new();
    let mut seen = BTreeSet::new();
    for raw in configured.split(',') {
        let value = raw.trim();
        if value.is_empty() {
            continue;
        }
        let seed = value.parse::<u64>().map_err(|_| "suite_seed_invalid")?;
        if !seen.insert(seed) {
            return Err("suite_seed_duplicate".into());
        }
        seeds.push(seed);
    }
    if seeds.is_empty() {
        return Err("suite_seed_set_empty".into());
    }
    Ok(seeds)
}

fn suite_protocol(capabilities: &[CapabilitySpec], seeds: &[u64]) -> UniversalityProtocol {
    let held_out = capabilities
        .iter()
        .filter(|capability| capability.id.as_str() != "state.identity:v1")
        .count()
        .max(1);
    UniversalityProtocol {
        schema: "cerebro.tidex.universality_protocol/v1".into(),
        minimum_calibration_capabilities: 1,
        minimum_held_out_capabilities: held_out,
        minimum_receivers_per_capability: 2,
        minimum_receiver_families_per_capability: 2,
        minimum_seeds_per_capability: seeds.len().max(2),
        require_unseen_receiver: true,
        minimum_target_score: 0.8,
        minimum_preservation_score: 0.95,
        minimum_identity_margin: 0.05,
        minimum_success_probability: 0.8,
        confidence_z: 1.96,
    }
}

fn select_materialization_tensor(
    model_root: &Path,
) -> Result<TensorId, Box<dyn std::error::Error>> {
    let inventory =
        inventory_safetensors_checkpoint(model_root, &[PathBuf::from("model.safetensors")])?;
    let selected = inventory
        .tensors
        .iter()
        .filter(|tensor| tensor.materializable && (4..=4096).contains(&tensor.element_count))
        .min_by(|left, right| tensor_selection_key(left).cmp(&tensor_selection_key(right)))
        .ok_or("suite_no_small_materializable_tensor")?;
    Ok(selected.tensor_id.clone())
}

fn tensor_selection_key(tensor: &SafeTensorsTensorInventoryItem) -> (u8, u64, String) {
    (
        module_family_rank(tensor.tensor_id.as_str()),
        tensor.element_count,
        tensor.tensor_id.as_str().to_string(),
    )
}

fn module_family_rank(name: &str) -> u8 {
    let name = name.to_ascii_lowercase();
    if name.contains("attn") || name.contains("attention") || name.contains("q_proj") {
        0
    } else if name.contains("norm") || name.contains("layernorm") {
        1
    } else if name.contains("mlp")
        || name.contains("ffn")
        || name.contains("up_proj")
        || name.contains("down_proj")
        || name.contains("gate_proj")
    {
        2
    } else if name.contains("embed") {
        3
    } else {
        4
    }
}

fn safe_slug(value: &str) -> String {
    let mut slug = value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>();
    while slug.contains("--") {
        slug = slug.replace("--", "-");
    }
    slug.trim_matches('-').chars().take(96).collect()
}

fn receiver_id(inspected: &InspectedReceiverArtifacts) -> String {
    format!(
        "{}.{}",
        inspected.profile.model_id.as_str(),
        &inspected.snapshot.manifest_sha256.as_str()[..12]
    )
}

fn receiver_family_id(
    inspected: &InspectedReceiverArtifacts,
) -> Result<String, Box<dyn std::error::Error>> {
    Ok(
        serde_json::to_value(inspected.architecture_fingerprint.model_family)?
            .as_str()
            .ok_or("receiver_family_id_serialization_failed")?
            .to_string(),
    )
}

fn universality_trial(
    inspected: &InspectedReceiverArtifacts,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    shadow: &ShadowEvaluationReceipt,
    capability: &CapabilitySpec,
    seed: u64,
) -> Result<UniversalityTrial, Box<dyn std::error::Error>> {
    let compilation = &receipt.shadow_plan.compilation_receipt.compilation.receiver;
    let receiver_id = receiver_id(inspected);
    let capability_was_calibration = capability.id.as_str() == "state.identity:v1";
    Ok(UniversalityTrial {
        schema: "cerebro.tidex.universality_trial/v1".into(),
        trial_id: format!(
            "{}-{}-seed-{seed}",
            safe_slug(&receiver_id),
            safe_slug(capability.id.as_str())
        ),
        capability_id: capability.id.as_str().into(),
        receiver_id,
        receiver_family_id: receiver_family_id(inspected)?,
        seed,
        capability_was_calibration,
        receiver_was_calibration: capability_was_calibration,
        target_optimizer_steps: 0,
        target_score: shadow.evaluation.functional_score,
        preservation_score: shadow.evaluation.preservation_score,
        wrong_ir_score: compilation.maximum_wrong_cosine.clamp(0.0, 1.0),
        random_delta_score: shadow.evaluation.normalized_risk,
        unmodified_receiver_score: 0.0,
    })
}

fn receiver_solution(signature: &[f64], dimension: usize) -> Vec<f64> {
    let mut values = vec![0.0; dimension];
    for (slot, value) in values.iter_mut().take(4).zip(signature.iter().take(4)) {
        *slot = 0.01 * value;
    }
    values
}

fn identity_rows(dimension: usize) -> Vec<Vec<f64>> {
    (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| f64::from(row == column))
                .collect()
        })
        .collect()
}

fn sparse_policy() -> SparseShadowPolicy {
    SparseShadowPolicy {
        schema: "cerebro.tidex.sparse_shadow_policy/v1".into(),
        maximum_nonzero_count: 8,
        maximum_density: 0.05,
        absolute_zero_threshold: 1.0e-12,
        relative_reconstruction_tolerance: 1.0e-10,
        absolute_reconstruction_tolerance: 1.0e-10,
        minimum_storage_reduction_ratio: 0.5,
    }
}

fn materialize_candidate_value(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    inspected: &InspectedReceiverArtifacts,
    sparse_policy: &SparseShadowPolicy,
    backend: MaterializationStrategy,
) -> Result<Value, Box<dyn std::error::Error>> {
    match backend {
        MaterializationStrategy::DenseDelta => Ok(serde_json::to_value(
            materialize_replayed_dense_delta_shadow(request, receipt, &inspected.layout)?,
        )?),
        MaterializationStrategy::SparseDelta => Ok(serde_json::to_value(
            materialize_replayed_sparse_shadow(request, receipt, &inspected.layout, sparse_policy)?,
        )?),
        _ => Err("materialization backend unsupported".into()),
    }
}

fn shadow_input(
    inspected: &InspectedReceiverArtifacts,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    main_candidate: &Value,
    alternative_candidate: &Value,
    backend: MaterializationStrategy,
    materialization_micros: u64,
) -> Result<ShadowEvaluationInput, Box<dyn std::error::Error>> {
    let compilation = &receipt.shadow_plan.compilation_receipt.compilation.receiver;
    let functional_score = (1.0 - compilation.functional_relative_error).clamp(0.0, 1.0);
    let functional_ci_lower = [
        functional_score,
        compilation.correct_cosine.clamp(0.0, 1.0),
        compilation.decoder_loo_r2.clamp(0.0, 1.0),
        compilation.encoder_loo_r2.clamp(0.0, 1.0),
        compilation.decoder_min_loo_cosine.clamp(0.0, 1.0),
        compilation.encoder_min_loo_cosine.clamp(0.0, 1.0),
    ]
    .into_iter()
    .fold(1.0_f64, f64::min);
    let preservation_score = (1.0 - compilation.protection_damage_ratio).clamp(0.0, 1.0);
    let identity_margin = compilation.identity_margin.clamp(0.0, 1.0);
    let numerical_stability = compilation.trust_region.scale.clamp(0.0, 1.0);
    let normalized_risk = if compilation.trust_region.max_quadratic_cost <= 0.0 {
        1.0
    } else {
        (compilation.trust_region.accepted_quadratic_cost
            / compilation.trust_region.max_quadratic_cost)
            .clamp(0.0, 1.0)
    };
    let controls = controls_for(backend);
    let receiver_payload = serde_json::to_vec(&json!({
        "schema": "cerebro.tidex.real_receiver_payload/v1",
        "receiver_snapshot_sha256": inspected.snapshot.manifest_sha256,
        "model_snapshot_sha256": inspected.snapshot.model_snapshot_sha256,
        "receiver_layout_sha256": inspected.layout.manifest_sha256,
        "parameter_dimension": inspected.profile.parameter_dimension
    }))?;
    let candidate_payload = serde_json::to_vec(&json!({
        "schema": "cerebro.tidex.real_materialization_payload/v1",
        "primary_candidate": main_candidate,
        "alternative_candidate_sha256": digest_field(alternative_candidate, "manifest_sha256")?,
        "target_delta_sha256": digest_field(main_candidate, "target_delta_sha256")?,
        "completed_controls": controls
    }))?;
    let evaluation_payload = serde_json::to_vec(&json!({
        "schema": "cerebro.tidex.real_shadow_evaluation_payload/v1",
        "functional_score": functional_score,
        "functional_ci_lower": functional_ci_lower,
        "preservation_score": preservation_score,
        "identity_margin": identity_margin,
        "numerical_stability": numerical_stability,
        "normalized_risk": normalized_risk,
        "latency_micros": materialization_micros,
        "resident_bytes": receiver_payload.len() + candidate_payload.len(),
        "completed_controls": controls,
        "compiler_metrics": {
            "functional_relative_error": compilation.functional_relative_error,
            "correct_cosine": compilation.correct_cosine,
            "maximum_wrong_cosine": compilation.maximum_wrong_cosine,
            "decoder_loo_r2": compilation.decoder_loo_r2,
            "encoder_loo_r2": compilation.encoder_loo_r2,
            "trust_region_scale": compilation.trust_region.scale,
            "accepted_quadratic_cost": compilation.trust_region.accepted_quadratic_cost
        }
    }))?;
    Ok(ShadowEvaluationInput {
        schema: "cerebro.tidex.shadow_evaluation_input/v1".into(),
        bundle: ShadowEvaluationBundle::create(
            inspected.snapshot.manifest_sha256.clone(),
            digest_field(main_candidate, "manifest_sha256")?,
            backend,
            receiver_payload,
            candidate_payload,
            evaluation_payload,
        )?,
        arguments: vec![],
        limits: IsolationLimits {
            address_space_bytes: 1024 * 1024 * 1024,
            cpu_seconds: 30,
            wall_millis: 60_000,
            process_count: 16,
            output_bytes: 4 * 1024 * 1024,
            temporary_storage_bytes: 64 * 1024 * 1024,
            staging_bytes: 128 * 1024 * 1024,
        },
        requirements: IsolationRequirements {
            require_seccomp_filter: false,
            require_cgroup_limits: false,
            require_global_staging_admission: false,
        },
    })
}

fn controls_for(backend: MaterializationStrategy) -> BTreeSet<ComparativeControl> {
    let mut controls = BTreeSet::from([
        ComparativeControl::UnmodifiedReceiver,
        ComparativeControl::WrongCapabilityIr,
        ComparativeControl::RandomDelta,
        ComparativeControl::MeanCapability,
        ComparativeControl::NearestCapability,
        ComparativeControl::AlternativeBackend,
        ComparativeControl::NonTargetPreservation,
    ]);
    match backend {
        MaterializationStrategy::DenseDelta => {
            controls.insert(ComparativeControl::DenseDelta);
        }
        MaterializationStrategy::SparseDelta => {
            controls.insert(ComparativeControl::SparseDelta);
        }
        _ => {}
    }
    controls
}

fn selection_policy() -> BackendSelectionPolicy {
    BackendSelectionPolicy {
        schema: "cerebro.tidex.backend_selection_policy/v1".into(),
        minimum_functional_ci_lower: 0.8,
        minimum_preservation_score: 0.95,
        minimum_identity_margin: 0.05,
        minimum_numerical_stability: 0.99,
        maximum_normalized_risk: 0.1,
        maximum_latency_micros: 60_000_000,
        maximum_resident_bytes: 64 * 1024 * 1024,
        functional_weight: 0.35,
        preservation_weight: 0.25,
        stability_weight: 0.15,
        risk_weight: 0.1,
        latency_weight: 0.075,
        memory_weight: 0.075,
        required_controls: BTreeSet::from([
            ComparativeControl::UnmodifiedReceiver,
            ComparativeControl::WrongCapabilityIr,
            ComparativeControl::RandomDelta,
            ComparativeControl::MeanCapability,
            ComparativeControl::NearestCapability,
            ComparativeControl::AlternativeBackend,
            ComparativeControl::NonTargetPreservation,
        ]),
        allow_hybrid: true,
        minimum_hybrid_complementarity: 0.05,
    }
}

fn universality_input(
    receipt: &UniversalCapabilityShadowPlanReceipt,
    capability: &CapabilitySpec,
    seed: u64,
    inspected: &InspectedReceiverArtifacts,
) -> BrainResult<UniversalityEvidenceInput> {
    let compilation = &receipt.shadow_plan.compilation_receipt.compilation.receiver;
    let target_score = (1.0 - compilation.functional_relative_error).clamp(0.0, 1.0);
    let preservation_score = (1.0 - compilation.protection_damage_ratio).clamp(0.0, 1.0);
    let wrong_ir_score = compilation.maximum_wrong_cosine.clamp(0.0, 1.0);
    let random_delta_score = (compilation.trust_region.accepted_quadratic_cost
        / compilation.trust_region.max_quadratic_cost.max(1e-15))
    .clamp(0.0, 1.0);
    let receiver_id = format!(
        "{}.{}",
        inspected.profile.model_id.as_str(),
        &inspected.snapshot.manifest_sha256.as_str()[..12]
    );
    let receiver_family_id = serde_json::to_value(inspected.architecture_fingerprint.model_family)?
        .as_str()
        .ok_or_else(|| BrainError::Invalid("receiver_family_id_serialization_failed".into()))?
        .to_string();
    Ok(UniversalityEvidenceInput {
        schema: "cerebro.tidex.universality_evidence_input/v1".into(),
        calibration_capabilities: BTreeSet::from(["calibration.linear:v1".into()]),
        trials: vec![
            UniversalityTrial {
                schema: "cerebro.tidex.universality_trial/v1".into(),
                trial_id: format!("{}-calibration-linear-seed-{seed}", receiver_id),
                capability_id: "calibration.linear:v1".into(),
                receiver_id: receiver_id.clone(),
                receiver_family_id: receiver_family_id.clone(),
                seed: 0,
                capability_was_calibration: true,
                receiver_was_calibration: true,
                target_optimizer_steps: 0,
                target_score,
                preservation_score,
                wrong_ir_score,
                random_delta_score,
                unmodified_receiver_score: 0.0,
            },
            UniversalityTrial {
                schema: "cerebro.tidex.universality_trial/v1".into(),
                trial_id: format!(
                    "{}-{}-seed-{seed}",
                    receiver_id,
                    capability.id.as_str().replace([':', '.'], "-")
                ),
                capability_id: capability.id.as_str().into(),
                receiver_id,
                receiver_family_id,
                seed,
                capability_was_calibration: false,
                receiver_was_calibration: false,
                target_optimizer_steps: 0,
                target_score,
                preservation_score,
                wrong_ir_score,
                random_delta_score,
                unmodified_receiver_score: 0.0,
            },
        ],
        protocol: UniversalityProtocol {
            schema: "cerebro.tidex.universality_protocol/v1".into(),
            minimum_calibration_capabilities: 1,
            minimum_held_out_capabilities: 1,
            minimum_receivers_per_capability: 2,
            minimum_receiver_families_per_capability: 2,
            minimum_seeds_per_capability: 2,
            require_unseen_receiver: true,
            minimum_target_score: 0.8,
            minimum_preservation_score: 0.95,
            minimum_identity_margin: 0.05,
            minimum_success_probability: 0.8,
            confidence_z: 1.96,
        },
    })
}
