use cerebro_tidex::acquisition_contract::{
    AcquisitionBudget, AcquisitionRequest, AcquisitionScope, DeclaredRelativePath, NoisePolicy,
    RequestedResidency,
};
use cerebro_tidex::activation_steering_materializer::{
    materialize_replayed_activation_steering_shadow, ActivationSteeringLayout,
    ActivationSteeringPolicy,
};
use cerebro_tidex::capability_discovery::CapabilityDiscoveryRequest;
use cerebro_tidex::checkpoint_adapter::{inspect_safetensors_receiver, SafeTensorsReceiverRequest};
use cerebro_tidex::content_vault::capture_to_vault;
use cerebro_tidex::dense_shadow_materializer::materialize_replayed_dense_delta_shadow;
use cerebro_tidex::identity::AcquisitionId;
use cerebro_tidex::isolated_execution::AuthenticatedBytes;
use cerebro_tidex::low_rank_shadow_materializer::{
    materialize_replayed_low_rank_shadow, LowRankShadowPolicy,
};
use cerebro_tidex::materialization_selector::BackendSelectionInput;
use cerebro_tidex::receiver_compiler::{
    benchmark_receiver_portability_leave_one_out, ReceiverPortabilityBenchmarkInput,
};
use cerebro_tidex::receiver_layout::ReceiverMaterializationLayout;
use cerebro_tidex::shadow_evaluation::{run_shadow_evaluation, ShadowEvaluationInput};
use cerebro_tidex::sparse_shadow_materializer::{
    materialize_replayed_sparse_shadow, SparseShadowPolicy,
};
use cerebro_tidex::universal_capability_compiler::{
    execute_experimental_universal_capability_request, execute_universal_capability_shadow_plan,
    replay_experimental_universal_capability_request, replay_universal_capability_shadow_plan,
    UniversalCapabilityCompilationReceipt, UniversalCapabilityCompilationRequest,
    UniversalCapabilityPlanningRequest, UniversalCapabilityShadowPlanReceipt,
};
use cerebro_tidex::universal_promotion_gate::{
    evaluate_universal_promotion_gate, UniversalPromotionGateRequest,
};
use cerebro_tidex::universality_evidence::UniversalityEvidenceInput;
use cerebro_tidex::workspace::{
    add_model, configured_tidex_home, create_workspace, current_workspace, load_model, use_model,
    use_workspace, ModelProfile, ModelProvider,
};
use serde_json::json;
use std::fs;
use std::io::Read;
use std::path::Path;

const MAX_BENCHMARK_INPUT_BYTES: u64 = 64 * 1024 * 1024;

fn main() {
    if let Err(error) = run(std::env::args().skip(1).collect()) {
        eprintln!("{error}");
        std::process::exit(2);
    }
}

fn run(args: Vec<String>) -> Result<(), Box<dyn std::error::Error>> {
    match args.as_slice() {
        [area, command, name, flag, target]
            if area == "workspace" && command == "create" && flag == "--target" =>
        {
            let home = configured_tidex_home()?;
            let manifest = create_workspace(&home, name, Path::new(target))?;
            println!("{}", serde_json::to_string_pretty(&manifest)?);
        }
        [area, command, name] if area == "workspace" && command == "use" => {
            let home = configured_tidex_home()?;
            use_workspace(&home, name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&current_workspace(&home)?)?
            );
        }
        [area, command] if area == "workspace" && command == "show" => {
            let home = configured_tidex_home()?;
            println!(
                "{}",
                serde_json::to_string_pretty(&current_workspace(&home)?)?
            );
        }
        [area, command, name, provider_flag, provider, endpoint_flag, endpoint, model_flag, model]
            if area == "model"
                && command == "add"
                && provider_flag == "--provider"
                && endpoint_flag == "--url"
                && model_flag == "--model" =>
        {
            let home = configured_tidex_home()?;
            let provider = match provider.as_str() {
                "openai-compatible" => ModelProvider::OpenAiCompatible,
                _ => return Err("model_provider_invalid".into()),
            };
            add_model(
                &home,
                ModelProfile {
                    schema: "cerebro.tidex.model_profile/v1".into(),
                    name: name.clone(),
                    provider,
                    endpoint: endpoint.clone(),
                    model: model.clone(),
                },
            )?;
            println!(
                "{}",
                serde_json::to_string_pretty(&load_model(&home, name)?)?
            );
        }
        [area, command, name] if area == "model" && command == "use" => {
            let home = configured_tidex_home()?;
            use_model(&home, name)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&load_model(&home, name)?)?
            );
        }
        [command] if command == "acquire" => {
            let home = configured_tidex_home()?;
            acquire_workspace(&home, None)?
        }
        [command, flag, path] if command == "acquire" && flag == "--path" => {
            let home = configured_tidex_home()?;
            acquire_workspace(&home, Some(Path::new(path)))?
        }
        [area, command, path] if area == "benchmark" && command == "portability" => {
            let input: ReceiverPortabilityBenchmarkInput = read_json_bounded(Path::new(path))?;
            let report = benchmark_receiver_portability_leave_one_out(&input)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, path] if area == "compile" && command == "universal" => {
            let request: UniversalCapabilityCompilationRequest =
                read_json_bounded(Path::new(path))?;
            let report = execute_experimental_universal_capability_request(&request)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        [area, command, request_path, receipt_path]
            if area == "compile" && command == "universal-replay" =>
        {
            let request: UniversalCapabilityCompilationRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityCompilationReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            replay_experimental_universal_capability_request(&request, &receipt)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema":"cerebro.tidex.universal_capability_compilation_replay/v1",
                    "request_sha256":receipt.request_sha256,
                    "replayed":true
                }))?
            );
        }
        [area, command, path] if area == "compile" && command == "universal-plan" => {
            let request: UniversalCapabilityPlanningRequest = read_json_bounded(Path::new(path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&execute_universal_capability_shadow_plan(&request)?)?
            );
        }
        [area, command, request_path, receipt_path]
            if area == "compile" && command == "universal-plan-replay" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            replay_universal_capability_shadow_plan(&request, &receipt)?;
            println!(
                "{}",
                serde_json::to_string_pretty(
                    &json!({"schema":"cerebro.tidex.universal_shadow_plan_replay/v1","planning_request_sha256":receipt.planning_request_sha256,"replayed":true})
                )?
            );
        }
        [area, backend, request_path, receipt_path, layout_path]
            if area == "materialize" && backend == "dense" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_dense_delta_shadow(
                    &request, &receipt, &layout
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, layout_path, policy_path]
            if area == "materialize" && backend == "low-rank" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            let policy: LowRankShadowPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_low_rank_shadow(
                    &request, &receipt, &layout, &policy
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, layout_path, policy_path]
            if area == "materialize" && backend == "sparse" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let layout: ReceiverMaterializationLayout = read_json_bounded(Path::new(layout_path))?;
            let policy: SparseShadowPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_sparse_shadow(
                    &request, &receipt, &layout, &policy
                )?)?
            );
        }
        [area, backend, request_path, receipt_path, receiver_layout_path, steering_layout_path, policy_path]
            if area == "materialize" && backend == "steering" =>
        {
            let request: UniversalCapabilityPlanningRequest =
                read_json_bounded(Path::new(request_path))?;
            let receipt: UniversalCapabilityShadowPlanReceipt =
                read_json_bounded(Path::new(receipt_path))?;
            let receiver_layout: ReceiverMaterializationLayout =
                read_json_bounded(Path::new(receiver_layout_path))?;
            let steering_layout: ActivationSteeringLayout =
                read_json_bounded(Path::new(steering_layout_path))?;
            let policy: ActivationSteeringPolicy = read_json_bounded(Path::new(policy_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&materialize_replayed_activation_steering_shadow(
                    &request,
                    &receipt,
                    &receiver_layout,
                    &steering_layout,
                    &policy
                )?)?
            );
        }
        [area, command, root, request_path]
            if area == "receiver" && command == "inspect-safetensors" =>
        {
            let request: SafeTensorsReceiverRequest = read_json_bounded(Path::new(request_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&inspect_safetensors_receiver(
                    Path::new(root),
                    &request
                )?)?
            );
        }
        [area, command, input_path] if area == "select" && command == "backend" => {
            let input: BackendSelectionInput = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "measure" && command == "universality" => {
            let input: UniversalityEvidenceInput = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "discover" && command == "capabilities" => {
            let input: CapabilityDiscoveryRequest = read_json_bounded(Path::new(input_path))?;
            println!("{}", serde_json::to_string_pretty(&input.execute()?)?);
        }
        [area, command, input_path] if area == "gate" && command == "promotion" => {
            let input: UniversalPromotionGateRequest = read_json_bounded(Path::new(input_path))?;
            println!(
                "{}",
                serde_json::to_string_pretty(&evaluate_universal_promotion_gate(&input)?)?
            );
        }
        [area, command, runner_path, input_path] if area == "shadow" && command == "run" => {
            let input: ShadowEvaluationInput = read_json_bounded(Path::new(input_path))?;
            if input.schema != "cerebro.tidex.shadow_evaluation_input/v1" {
                return Err("shadow_evaluation_input_schema_invalid".into());
            }
            let runner = AuthenticatedBytes::from_trusted_bytes(read_bytes_bounded(
                Path::new(runner_path),
                64 * 1024 * 1024,
            )?);
            println!(
                "{}",
                serde_json::to_string_pretty(&run_shadow_evaluation(
                    runner,
                    &input.bundle,
                    input.arguments,
                    input.limits,
                    input.requirements
                )?)?
            );
        }
        [command] if command == "capabilities" => {
            let home = configured_tidex_home()?;
            let workspace = current_workspace(&home)?;
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "schema":"cerebro.tidex.capabilities/v1",
                    "workspace":workspace.name,
                    "target":workspace.target,
                    "capabilities":[
                        {"id":"acquisition.capture","status":"implemented","engine":"content_vault::capture_to_vault"},
                        {"id":"analysis.skill_fields","status":"implemented","engine":"BrainEngine::analyze"},
                        {"id":"learning.adaptive","status":"implemented","engine":"learning_orchestrator"},
                        {"id":"controller.learned","status":"implemented","engine":"learned_controller"},
                        {"id":"transport.functional","status":"implemented","engine":"transport::learn_functional_transplant"},
                        {"id":"transport.relational","status":"implemented","engine":"transport::learn_relational_transport"},
                        {"id":"compile.skill_fields","status":"implemented","engine":"parametric_program::compile_operator_to_fields"},
                        {"id":"compile.receiver","status":"implemented_experimental","engine":"receiver_compiler::compile_receiver_capability","evidence_status":"bounded_cross_model_experimental","reason":"receiver-native functional capability compilation is implemented with held-out functional verification, protection and trust-region gates; V66 adds one-seed cross-model evidence for Qwen2.5-Coder-1.5B -> SmolLM2-1.7B on MBPP using frozen-backbone LoRA materialization. This does not establish universal portability, zero-optimization translation, or generality across model and capability families"},
                        {"id":"compile.universal_capability","status":"implemented_experimental","engine":"universal_capability_compiler::execute_experimental_universal_capability_request","reason":"versioned JSON request binds a verified source envelope, closed CapabilityIR, operational contract, receiver calibration, protection and risk controls. The emitted receipt is request-bound and replayable. A passing result is experimental only and cannot authorize deployment or promotion"},
                        {"id":"receiver.inspect_safetensors","status":"implemented_experimental","engine":"checkpoint_adapter::inspect_safetensors_receiver","reason":"read-only inspection authenticates exact checkpoint, configuration and tokenizer bytes; validates SafeTensors geometry and produces snapshot-bound receiver layout plus architecture-family fingerprint"},
                        {"id":"capability.discover_behavior","status":"implemented_experimental","engine":"capability_discovery::discover_capabilities","reason":"repeated functional probes, seeds, controls, closure and contraction determine whether evidence is ready for CapabilityIR; tensor names alone never declare a capability"},
                        {"id":"materialize.shadow","status":"implemented_experimental","backends":["receiver_coordinates","dense_delta","low_rank","sparse_delta","activation_steering"],"reason":"all backends consume only replayed compiler target deltas and remain inert, layout-bound and non-actuating"},
                        {"id":"materialize.select","status":"implemented_experimental","engine":"materialization_selector::select_materialization_backend","reason":"comparative gates, Pareto ranking and evidence-backed hybrid composition; no backend is privileged by default"},
                        {"id":"measure.universality_n","status":"implemented_experimental","engine":"universality_evidence::measure_universality_n","reason":"N is computed only from held-out zero-step evidence with receiver, family, seed, confidence, preservation and negative-control gates"},
                        {"id":"gate.universal_promotion","status":"implemented_non_actuating","engine":"universal_promotion_gate::evaluate_universal_promotion_gate","reason":"can establish readiness for a separate promotion authority but always returns authorizes_activation=false"},
                        {"id":"benchmark.portability","canonical_name":"benchmark.receiver_compilation","status":"implemented","engine":"receiver_compiler::benchmark_receiver_portability_leave_one_out","reason":"leave-one-capability-out functional-space benchmark over declared calibration cases; the legacy portability id is retained for compatibility, while the benchmark measures receiver compilation recovery inside that domain and is not a universal cross-model claim"},
                        {"id":"runtime.sleep","status":"implemented","engine":"BrainEngine::sleep_cycle"},
                        {"id":"capability_ir.v63.contract","status":"implemented_foundation","engine":"capability_ir::OperationalCapabilityContract","reason":"StateIR anchors, repeated OperatorIR transitions, canonical transition signatures, closure and contraction verification are implemented; evidence is bounded to tested domains and does not establish a universal capability representation across arbitrary models or tasks"},
                        {"id":"model.assistance","status":"configured_not_authoritative","reason":"model profiles are selectable; no model call is permitted to create evidence or promotion authority"}
                    ]
                }))?
            );
        }
        _ => return Err(usage().into()),
    }
    Ok(())
}

fn read_json_bounded<T: serde::de::DeserializeOwned>(
    path: &Path,
) -> Result<T, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(MAX_BENCHMARK_INPUT_BYTES.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > MAX_BENCHMARK_INPUT_BYTES {
        return Err("tidex_benchmark_input_too_large".into());
    }
    Ok(serde_json::from_slice(&bytes)?)
}

fn read_bytes_bounded(path: &Path, maximum: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    let mut bytes = Vec::new();
    file.take(maximum.saturating_add(1))
        .read_to_end(&mut bytes)?;
    if u64::try_from(bytes.len())? > maximum {
        return Err("tidex_input_too_large".into());
    }
    Ok(bytes)
}

fn acquire_workspace(
    home: &Path,
    selected: Option<&Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let workspace = current_workspace(home)?;
    let private_root = workspace.private_root(home);
    let scope = match selected {
        None => AcquisitionScope::WholeProject,
        Some(path) => AcquisitionScope::DeclaredPaths {
            roots: vec![DeclaredRelativePath::parse(path.to_path_buf())?],
        },
    };
    let request = AcquisitionRequest::new(
        AcquisitionId::parse(format!("workspace-{}-capture", workspace.name))?,
        scope,
        RequestedResidency::BestVerified,
        NoisePolicy::ConservativeGeneratedArtifacts,
        AcquisitionBudget {
            max_files: 100_000,
            max_total_bytes: 8 * 1024 * 1024 * 1024,
        },
        vec![],
    )?;
    let receipt = capture_to_vault(&workspace.target, &private_root, &request)?;
    let reference = receipt.persist(&private_root)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "schema":"cerebro.tidex.workspace_acquisition/v1",
            "workspace":workspace.name,
            "target":workspace.target,
            "capture_receipt_sha256":receipt.manifest_sha256(),
            "capture_receipt":reference,
            "system_envelope_sha256":receipt.envelope().manifest_sha256(),
            "completeness":receipt.envelope().completeness(),
            "entries":receipt.envelope().entries().len(),
            "bytes":receipt.total_file_bytes()
        }))?
    );
    Ok(())
}

fn usage() -> &'static str {
    "usage:\n  tidex workspace create <name> --target <absolute-path>\n  tidex workspace use <name>\n  tidex workspace show\n  tidex model add <name> --provider openai-compatible --url <endpoint> --model <model>\n  tidex model use <name>\n  tidex acquire [--path <relative-project-path>]\n  tidex benchmark portability <input.json>\n  tidex compile universal <input.json>\n  tidex compile universal-replay <input.json> <receipt.json>\n  tidex compile universal-plan <request.json>\n  tidex compile universal-plan-replay <request.json> <receipt.json>\n  tidex materialize dense <request.json> <receipt.json> <layout.json>\n  tidex materialize low-rank <request.json> <receipt.json> <layout.json> <policy.json>\n  tidex materialize sparse <request.json> <receipt.json> <layout.json> <policy.json>\n  tidex materialize steering <request.json> <receipt.json> <receiver-layout.json> <steering-layout.json> <policy.json>\n  tidex receiver inspect-safetensors <root> <request.json>\n  tidex discover capabilities <input.json>\n  tidex shadow run <runner> <input.json>\n  tidex select backend <input.json>\n  tidex measure universality <input.json>\n  tidex gate promotion <input.json>\n  tidex capabilities"
}
