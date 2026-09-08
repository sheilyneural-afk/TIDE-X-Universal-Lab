//! Experimental orchestration for receiver-native capability compilation.
//!
//! This module intentionally contains no donor-weight, adapter, LoRA, or task
//! vector representation.  It binds the existing authenticated structural IR
//! and operational contract to the existing receiver compiler.  Consequently
//! a successful result is evidence only for the supplied calibration domain;
//! it is never an automatic residency or promotion decision.

use crate::acquisition_contract::SystemEnvelope;
use crate::capability_ir::{CapabilityIr, OperationalCapabilityContract};
use crate::contracts::ProtectedCortex;
use crate::digest::{CapabilityIrDigest, Sha256Digest, SystemEnvelopeDigest};
use crate::error::{BrainError, BrainResult};
use crate::linalg::Matrix;
use crate::receiver_compiler::{
    compile_receiver_capability, ReceiverCalibrationSet, ReceiverCompilation,
    ReceiverCompilerPolicy,
};
use crate::receiver_profile::{
    assess_compatibility, create_shadow_plan, CapabilityRequirements, CompatibilityAssessment,
    MaterializationPlan, MaterializationStrategy, ReceiverProfile,
};
use serde::{Deserialize, Serialize};

/// Portable input envelope for one experimental receiver compilation.
///
/// `risk_metric_rows` is used instead of serialising the internal matrix type.
/// It must describe a finite square matrix with one row per receiver parameter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilationRequest {
    pub schema: String,
    pub system_envelope: SystemEnvelope,
    pub capability_ir: CapabilityIr,
    pub operational_contract: OperationalCapabilityContract,
    pub calibration: ReceiverCalibrationSet,
    pub protected_cortex: ProtectedCortex,
    pub risk_metric_rows: Vec<Vec<f64>>,
    pub policy: ReceiverCompilerPolicy,
}

/// The only dispositions this experimental boundary can produce.
///
/// `ExperimentalOnly` deliberately means that all local gates passed. It is
/// not a portability, equivalence, deployment, or promotion assertion.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UniversalCapabilityDisposition {
    ExperimentalOnly,
    Rejected,
}

/// A provenance-bound result from one receiver-native compilation attempt.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilation {
    pub schema: String,
    pub source_envelope_sha256: SystemEnvelopeDigest,
    pub capability_ir_sha256: CapabilityIrDigest,
    pub receiver: ReceiverCompilation,
    pub disposition: UniversalCapabilityDisposition,
}

/// A replayable, request-bound record of one experimental compilation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityCompilationReceipt {
    pub schema: String,
    pub request_sha256: Sha256Digest,
    pub compilation: UniversalCapabilityCompilation,
}

/// A non-actuating composition of compilation, compatibility assessment and
/// materialization planning.  The resulting plan remains shadow-only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityPlanningRequest {
    pub schema: String,
    pub compilation: UniversalCapabilityCompilationRequest,
    pub receiver_profile: ReceiverProfile,
    pub capability_requirements: CapabilityRequirements,
    pub requested_strategy: MaterializationStrategy,
    pub affected_regions: Vec<crate::identity::TensorId>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityShadowPlan {
    pub schema: String,
    pub compilation_receipt: UniversalCapabilityCompilationReceipt,
    pub compatibility: CompatibilityAssessment,
    pub materialization_plan: MaterializationPlan,
}

/// A replayable record for the complete shadow-planning decision.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalCapabilityShadowPlanReceipt {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub shadow_plan: UniversalCapabilityShadowPlan,
}

impl UniversalCapabilityCompilation {
    pub fn is_experimentally_usable(&self) -> bool {
        self.disposition == UniversalCapabilityDisposition::ExperimentalOnly
    }
}

/// Compile a sealed capability into receiver-native parameters for a declared
/// experimental calibration domain.
///
/// The function accepts semantic representations and receiver calibration
/// data, never donor parameters. It verifies the source envelope and IR chain
/// before delegating all numerical, protected-subspace, trust-region, and
/// operational checks to [`compile_receiver_capability`].
pub fn compile_experimental_universal_capability(
    envelope: &SystemEnvelope,
    ir: &CapabilityIr,
    operational: &OperationalCapabilityContract,
    calibration: &ReceiverCalibrationSet,
    protected_cortex: &ProtectedCortex,
    risk_metric: &Matrix,
    policy: &ReceiverCompilerPolicy,
) -> BrainResult<UniversalCapabilityCompilation> {
    envelope.verify_manifest()?;
    ir.validate_against(envelope)?;
    operational.validate_against(ir)?;

    let receiver = compile_receiver_capability(
        ir,
        operational,
        calibration,
        protected_cortex,
        risk_metric,
        policy,
    )?;
    let disposition = if receiver.allowed && receiver.operational_verification.allowed {
        UniversalCapabilityDisposition::ExperimentalOnly
    } else {
        UniversalCapabilityDisposition::Rejected
    };

    Ok(UniversalCapabilityCompilation {
        schema: "cerebro.tidex.universal_capability_compilation/v1".into(),
        source_envelope_sha256: envelope.manifest_sha256().clone(),
        capability_ir_sha256: ir.manifest_digest().clone(),
        receiver,
        disposition,
    })
}

/// Compile an independently serialised request.
///
/// This is the CLI and artifact boundary. It keeps the matrix wire format
/// explicit and rejects malformed rows before the numerical compiler sees it.
pub fn compile_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
) -> BrainResult<UniversalCapabilityCompilation> {
    if request.schema != "cerebro.tidex.universal_capability_compilation_request/v1" {
        return Err(crate::error::BrainError::Invalid(
            "universal_capability_compilation_request_schema".into(),
        ));
    }
    let risk_metric = Matrix::from_rows(&request.risk_metric_rows)?;
    if risk_metric.row_count() == 0 || risk_metric.row_count() != risk_metric.column_count() {
        return Err(crate::error::BrainError::Invalid(
            "universal_capability_compilation_risk_metric_shape".into(),
        ));
    }
    compile_experimental_universal_capability(
        &request.system_envelope,
        &request.capability_ir,
        &request.operational_contract,
        &request.calibration,
        &request.protected_cortex,
        &risk_metric,
        &request.policy,
    )
}

fn request_digest(request: &UniversalCapabilityCompilationRequest) -> BrainResult<Sha256Digest> {
    let payload = serde_json::to_vec(request)?;
    let mut framed = b"CEREBRO:TIDEX:UNIVERSAL-CAPABILITY-COMPILATION-REQUEST:v1\0".to_vec();
    framed.extend_from_slice(&payload);
    Ok(Sha256Digest::digest_bytes(&framed))
}

/// Execute a request and bind its exact wire representation to the result.
pub fn execute_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
) -> BrainResult<UniversalCapabilityCompilationReceipt> {
    Ok(UniversalCapabilityCompilationReceipt {
        schema: "cerebro.tidex.universal_capability_compilation_receipt/v1".into(),
        request_sha256: request_digest(request)?,
        compilation: compile_experimental_universal_capability_request(request)?,
    })
}

/// Recompute a receipt from its request and fail closed on any divergence.
pub fn replay_experimental_universal_capability_request(
    request: &UniversalCapabilityCompilationRequest,
    receipt: &UniversalCapabilityCompilationReceipt,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.universal_capability_compilation_receipt/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_compilation_receipt_schema".into(),
        ));
    }
    if receipt.request_sha256 != request_digest(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_request_digest_mismatch".into(),
        ));
    }
    let replay = compile_experimental_universal_capability_request(request)?;
    if replay != receipt.compilation {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_replay_mismatch".into(),
        ));
    }
    Ok(())
}

/// Produce a complete shadow-only plan. Rejected compilations are never
/// converted into plans, even if the receiver is otherwise compatible.
pub fn compile_and_plan_experimental_universal_capability(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<UniversalCapabilityShadowPlan> {
    if request.schema != "cerebro.tidex.universal_capability_planning_request/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_planning_request_schema".into(),
        ));
    }
    let compilation_receipt =
        execute_experimental_universal_capability_request(&request.compilation)?;
    if !compilation_receipt.compilation.is_experimentally_usable() {
        return Err(BrainError::Integrity(
            "universal_capability_compilation_not_usable".into(),
        ));
    }
    request
        .capability_requirements
        .validate_against(&request.compilation.capability_ir)?;
    if u64::try_from(
        compilation_receipt
            .compilation
            .receiver
            .receiver_parameter_dimension,
    )
    .map_err(|_| BrainError::Invalid("receiver_parameter_dimension_overflow".into()))?
        != request.receiver_profile.parameter_dimension
    {
        return Err(BrainError::Integrity(
            "receiver_profile_compilation_dimension_mismatch".into(),
        ));
    }
    let compatibility =
        assess_compatibility(&request.receiver_profile, &request.capability_requirements)?;
    let materialization_plan = create_shadow_plan(
        &request.receiver_profile,
        &compatibility,
        &request.capability_requirements,
        compilation_receipt.request_sha256.clone(),
        request.requested_strategy,
        request.affected_regions.clone(),
    )?;
    Ok(UniversalCapabilityShadowPlan {
        schema: "cerebro.tidex.universal_capability_shadow_plan/v1".into(),
        compilation_receipt,
        compatibility,
        materialization_plan,
    })
}

fn planning_request_digest(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSAL-CAPABILITY-PLANNING-REQUEST:v1\0",
        &serde_json::to_vec(request)?,
    ))
}

pub fn execute_universal_capability_shadow_plan(
    request: &UniversalCapabilityPlanningRequest,
) -> BrainResult<UniversalCapabilityShadowPlanReceipt> {
    Ok(UniversalCapabilityShadowPlanReceipt {
        schema: "cerebro.tidex.universal_capability_shadow_plan_receipt/v1".into(),
        planning_request_sha256: planning_request_digest(request)?,
        shadow_plan: compile_and_plan_experimental_universal_capability(request)?,
    })
}

pub fn replay_universal_capability_shadow_plan(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
) -> BrainResult<()> {
    if receipt.schema != "cerebro.tidex.universal_capability_shadow_plan_receipt/v1" {
        return Err(BrainError::Invalid(
            "universal_capability_shadow_plan_receipt_schema".into(),
        ));
    }
    if receipt.planning_request_sha256 != planning_request_digest(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_shadow_plan_request_digest_mismatch".into(),
        ));
    }
    if receipt.shadow_plan != compile_and_plan_experimental_universal_capability(request)? {
        return Err(BrainError::Integrity(
            "universal_capability_shadow_plan_replay_mismatch".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acquisition_contract::{
        AcquisitionBudget, AcquisitionRequest, AcquisitionScope, NoisePolicy, RequestedResidency,
    };
    use crate::capability_ir::{
        IrNode, OperatorIrTransition, OutputBinding, PrimitiveSet, StateIrAnchor, TypedPort,
        ValueReference,
    };
    use crate::identity::{AcquisitionId, CapabilityId, CapabilityNodeId, PortId, PrimitiveId};
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn fixture() -> (
        PathBuf,
        SystemEnvelope,
        CapabilityIr,
        OperationalCapabilityContract,
    ) {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("tidex-ucc-{}-{nonce}", std::process::id()));
        fs::create_dir_all(root.join("src")).unwrap();
        fs::write(root.join("src/capability.rs"), b"pub fn capability() {}\n").unwrap();
        let request = AcquisitionRequest::new(
            AcquisitionId::parse("ucc-fixture").unwrap(),
            AcquisitionScope::WholeProject,
            RequestedResidency::BestVerified,
            NoisePolicy::ExplicitOnly,
            AcquisitionBudget {
                max_files: 8,
                max_total_bytes: 1 << 20,
            },
            vec![],
        )
        .unwrap();
        let envelope = SystemEnvelope::capture(&root, &request).unwrap();
        let ir = CapabilityIr::new(
            CapabilityId::parse("state.toggle:v1").unwrap(),
            &envelope,
            PrimitiveSet::tidex_core_v1().unwrap(),
            vec![TypedPort::tensor_f64(PortId::parse("state").unwrap(), vec![2, 1]).unwrap()],
            vec![IrNode::new(
                CapabilityNodeId::parse("node.normalize").unwrap(),
                PrimitiveId::parse("tensor.normalize").unwrap(),
                vec![ValueReference::Input {
                    name: PortId::parse("state").unwrap(),
                }],
                TypedPort::tensor_f64(PortId::parse("normalized").unwrap(), vec![2, 1]).unwrap(),
                vec![PathBuf::from("src/capability.rs")],
            )
            .unwrap()],
            vec![OutputBinding::new(
                TypedPort::tensor_f64(PortId::parse("result").unwrap(), vec![2, 1]).unwrap(),
                ValueReference::NodeOutput {
                    node_id: CapabilityNodeId::parse("node.normalize").unwrap(),
                },
            )
            .unwrap()],
        )
        .unwrap();
        let pre = 2.0_f64.sqrt();
        let operational = OperationalCapabilityContract {
            schema: "cerebro.tidex.operational_capability/v1".into(),
            capability_id: ir.capability_id().clone(),
            capability_ir_sha256: ir.manifest_digest().clone(),
            state_dimension: 2,
            anchors: vec![
                StateIrAnchor {
                    anchor_id: "s0".into(),
                    state: vec![1.0, 0.0],
                },
                StateIrAnchor {
                    anchor_id: "s1".into(),
                    state: vec![0.0, 1.0],
                },
            ],
            transitions: vec![
                OperatorIrTransition {
                    operator_id: "toggle".into(),
                    source_anchor_id: "s0".into(),
                    target_anchor_id: "s1".into(),
                    observed_next_state: vec![0.0, 1.0],
                    pre_target_error: pre,
                    post_target_error: 0.0,
                },
                OperatorIrTransition {
                    operator_id: "toggle".into(),
                    source_anchor_id: "s1".into(),
                    target_anchor_id: "s0".into(),
                    observed_next_state: vec![1.0, 0.0],
                    pre_target_error: pre,
                    post_target_error: 0.0,
                },
            ],
            maximum_closure_error: 1e-5,
            maximum_contraction_ratio: 1e-5,
        };
        (root, envelope, ir, operational)
    }

    fn calibration() -> Vec<Vec<f64>> {
        vec![
            vec![1.0, 0.0, 0.0, 1.0],
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 1.0],
            vec![1.0, 0.5, 0.5, 1.0],
            vec![0.2, 1.0, 1.0, 0.2],
            vec![1.2, -0.2, 0.4, 0.8],
        ]
    }

    fn receiver_solution(functional: &[f64]) -> Vec<f64> {
        vec![
            2.0 * functional[0] + functional[1] - 0.5 * functional[2] + 0.1,
            -functional[0] + 1.5 * functional[2] + functional[3] - 0.2,
            0.5 * functional[1] + 2.0 * functional[3] + 0.3,
            functional[0] - functional[1] + functional[2] - functional[3] + 0.4,
            0.7 * functional[0] + 0.2 * functional[1] + 0.3 * functional[2] + 0.9 * functional[3]
                - 0.1,
        ]
    }

    #[test]
    fn binds_existing_verified_components_without_donor_parameters() {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let compilation = compile_experimental_universal_capability(
            &envelope,
            &ir,
            &operational,
            &ReceiverCalibrationSet {
                functional_signatures: functional.clone(),
                receiver_solutions: functional
                    .iter()
                    .map(|row| receiver_solution(row))
                    .collect(),
                wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
            },
            &ProtectedCortex {
                parameter_importance: vec![0.0; 5],
                directions: Vec::new(),
                max_damage_ratio: 0.01,
            },
            &Matrix::identity(5),
            &ReceiverCompilerPolicy {
                schema: "cerebro.tidex.receiver_compiler_policy/v1".into(),
                ridge: 1e-10,
                minimum_decoder_loo_r2: 0.999,
                minimum_encoder_loo_r2: 0.999,
                minimum_decoder_loo_cosine: 0.999,
                maximum_functional_relative_error: 1e-4,
                minimum_identity_margin: 0.05,
                maximum_quadratic_cost: 1e6,
            },
        )
        .unwrap();
        assert!(compilation.is_experimentally_usable(), "{compilation:#?}");
        assert_eq!(
            compilation.source_envelope_sha256,
            *envelope.manifest_sha256()
        );
        assert_eq!(compilation.capability_ir_sha256, *ir.manifest_digest());
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn serialized_request_is_executable_and_rejects_a_nonsquare_risk_metric() {
        let (root, envelope, ir, operational) = fixture();
        let functional = calibration();
        let request = UniversalCapabilityCompilationRequest {
            schema: "cerebro.tidex.universal_capability_compilation_request/v1".into(),
            system_envelope: envelope,
            capability_ir: ir,
            operational_contract: operational,
            calibration: ReceiverCalibrationSet {
                functional_signatures: functional.clone(),
                receiver_solutions: functional
                    .iter()
                    .map(|row| receiver_solution(row))
                    .collect(),
                wrong_functional_signatures: vec![functional[0].clone(), functional[2].clone()],
            },
            protected_cortex: ProtectedCortex {
                parameter_importance: vec![0.0; 5],
                directions: Vec::new(),
                max_damage_ratio: 0.01,
            },
            risk_metric_rows: (0..5)
                .map(|row| (0..5).map(|column| f64::from(row == column)).collect())
                .collect(),
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
        };
        let restored: UniversalCapabilityCompilationRequest =
            serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
        assert!(compile_experimental_universal_capability_request(&restored)
            .unwrap()
            .is_experimentally_usable());
        let receipt = execute_experimental_universal_capability_request(&restored).unwrap();
        replay_experimental_universal_capability_request(&restored, &receipt).unwrap();

        let mut altered_receipt = receipt.clone();
        altered_receipt.compilation.disposition = UniversalCapabilityDisposition::Rejected;
        assert!(
            replay_experimental_universal_capability_request(&restored, &altered_receipt).is_err()
        );

        let mut malformed = restored;
        malformed.risk_metric_rows.pop();
        assert!(compile_experimental_universal_capability_request(&malformed).is_err());
        fs::remove_dir_all(root).unwrap();
    }
}
