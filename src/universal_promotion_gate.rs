//! Final evidence gate. A passing result is readiness for a separate promotion
//! authority; this module has no activation or model-writing capability.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::materialization_selector::{BackendSelectionInput, BackendSelectionReceipt};
use crate::shadow_evaluation::ShadowEvaluationReceipt;
use crate::universality_evidence::{UniversalityEvidenceInput, UniversalityEvidenceReceipt};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalPromotionPolicy {
    pub schema: String,
    pub minimum_universality_n: usize,
    pub minimum_global_wilson_lower_bound: f64,
    pub require_all_selected_candidates_evaluated: bool,
}

impl UniversalPromotionPolicy {
    fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.universal_promotion_policy/v1"
            || self.minimum_universality_n == 0
            || !self.minimum_global_wilson_lower_bound.is_finite()
            || !(0.0..=1.0).contains(&self.minimum_global_wilson_lower_bound)
            || !self.require_all_selected_candidates_evaluated
        {
            return Err(BrainError::Invalid(
                "universal_promotion_policy_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum UniversalPromotionBlocker {
    UniversalityNInsufficient,
    GlobalConfidenceInsufficient,
    SelectedCandidateMissingEvaluation,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UniversalPromotionReadiness {
    ReadyForSeparatePromotionAuthority,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct UniversalPromotionGateRequest {
    pub schema: String,
    pub selection_input: BackendSelectionInput,
    pub selection_receipt: BackendSelectionReceipt,
    pub universality_input: UniversalityEvidenceInput,
    pub universality_receipt: UniversalityEvidenceReceipt,
    pub shadow_evaluations: Vec<ShadowEvaluationReceipt>,
    pub policy: UniversalPromotionPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct UniversalPromotionGateReceipt {
    pub schema: String,
    pub selection_sha256: Sha256Digest,
    pub universality_sha256: Sha256Digest,
    pub shadow_evaluations_sha256: Sha256Digest,
    pub policy_sha256: Sha256Digest,
    pub readiness: UniversalPromotionReadiness,
    pub blockers: BTreeSet<UniversalPromotionBlocker>,
    pub authorizes_activation: bool,
    pub manifest_sha256: Sha256Digest,
}

pub fn evaluate_universal_promotion_gate(
    request: &UniversalPromotionGateRequest,
) -> BrainResult<UniversalPromotionGateReceipt> {
    if request.schema != "cerebro.tidex.universal_promotion_gate_request/v1" {
        return Err(BrainError::Invalid(
            "universal_promotion_gate_request_invalid".into(),
        ));
    }
    request.policy.validate()?;
    request
        .selection_receipt
        .validate_against(&request.selection_input)?;
    request
        .universality_receipt
        .validate_against(&request.universality_input)?;
    let mut evaluated = BTreeSet::new();
    for receipt in &request.shadow_evaluations {
        receipt.validate_integrity()?;
        if !evaluated.insert(&receipt.evaluation.candidate_sha256) {
            return Err(BrainError::Invalid(
                "shadow_evaluation_candidate_duplicate".into(),
            ));
        }
    }
    let mut blockers = BTreeSet::new();
    if request.universality_receipt.universality_n < request.policy.minimum_universality_n {
        blockers.insert(UniversalPromotionBlocker::UniversalityNInsufficient);
    }
    if request.universality_receipt.global_wilson_lower_bound
        < request.policy.minimum_global_wilson_lower_bound
    {
        blockers.insert(UniversalPromotionBlocker::GlobalConfidenceInsufficient);
    }
    if request
        .selection_receipt
        .selected_candidates
        .iter()
        .any(|candidate| !evaluated.contains(candidate))
    {
        blockers.insert(UniversalPromotionBlocker::SelectedCandidateMissingEvaluation);
    }
    let readiness = if blockers.is_empty() {
        UniversalPromotionReadiness::ReadyForSeparatePromotionAuthority
    } else {
        UniversalPromotionReadiness::Rejected
    };
    let selection_sha256 = request.selection_receipt.manifest_sha256.clone();
    let universality_sha256 = request.universality_receipt.manifest_sha256.clone();
    let shadow_evaluations_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-EVALUATION-SET:v1\0",
        &serde_json::to_vec(&request.shadow_evaluations)?,
    );
    let policy_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSAL-PROMOTION-POLICY:v1\0",
        &serde_json::to_vec(&request.policy)?,
    );
    let mut receipt = UniversalPromotionGateReceipt {
        schema: "cerebro.tidex.universal_promotion_gate_receipt/v1".into(),
        selection_sha256,
        universality_sha256,
        shadow_evaluations_sha256,
        policy_sha256,
        readiness,
        blockers,
        authorizes_activation: false,
        manifest_sha256: Sha256Digest::zero(),
    };
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    receipt.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:UNIVERSAL-PROMOTION-GATE-RECEIPT:v1\0",
        &serde_json::to_vec(&unsigned)?,
    );
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materialization_selector::{
        BackendEvaluation, BackendSelectionPolicy, PairwiseComplementarity,
    };
    use crate::receiver_profile::MaterializationStrategy;
    use crate::universality_evidence::{UniversalityProtocol, UniversalityTrial};

    #[test]
    fn passing_statistics_cannot_bypass_missing_shadow_evaluation_or_activate() {
        let candidate = Sha256Digest::digest_bytes(b"candidate");
        let controls = BackendSelectionPolicy::rigorous_default(100, 100).required_controls;
        let selection_input = BackendSelectionInput {
            schema: "cerebro.tidex.backend_selection_input/v1".into(),
            evaluations: vec![BackendEvaluation {
                schema: "cerebro.tidex.backend_evaluation/v1".into(),
                candidate_sha256: candidate,
                strategy: MaterializationStrategy::SparseDelta,
                functional_score: 1.0,
                functional_ci_lower: 1.0,
                preservation_score: 1.0,
                identity_margin: 1.0,
                numerical_stability: 1.0,
                normalized_risk: 0.0,
                latency_micros: 1,
                resident_bytes: 1,
                completed_controls: controls,
            }],
            complementarity: Vec::<PairwiseComplementarity>::new(),
            policy: BackendSelectionPolicy::rigorous_default(100, 100),
        };
        let selection_receipt = selection_input.execute().unwrap();
        let universality_input = UniversalityEvidenceInput {
            schema: "cerebro.tidex.universality_evidence_input/v1".into(),
            calibration_capabilities: BTreeSet::from(["calibration:v1".into()]),
            trials: vec![
                UniversalityTrial {
                    schema: "cerebro.tidex.universality_trial/v1".into(),
                    trial_id: "calibration-trial".into(),
                    capability_id: "calibration:v1".into(),
                    receiver_id: "receiver.seen".into(),
                    receiver_family_id: "family.seen".into(),
                    seed: 0,
                    capability_was_calibration: true,
                    receiver_was_calibration: true,
                    target_optimizer_steps: 0,
                    target_score: 1.0,
                    preservation_score: 1.0,
                    wrong_ir_score: 0.0,
                    random_delta_score: 0.0,
                    unmodified_receiver_score: 0.0,
                },
                UniversalityTrial {
                    schema: "cerebro.tidex.universality_trial/v1".into(),
                    trial_id: "held-out-trial".into(),
                    capability_id: "held-out:v1".into(),
                    receiver_id: "receiver.new".into(),
                    receiver_family_id: "family.new".into(),
                    seed: 1,
                    capability_was_calibration: false,
                    receiver_was_calibration: false,
                    target_optimizer_steps: 0,
                    target_score: 1.0,
                    preservation_score: 1.0,
                    wrong_ir_score: 0.0,
                    random_delta_score: 0.0,
                    unmodified_receiver_score: 0.0,
                },
            ],
            protocol: UniversalityProtocol {
                schema: "cerebro.tidex.universality_protocol/v1".into(),
                minimum_calibration_capabilities: 1,
                minimum_held_out_capabilities: 1,
                minimum_receivers_per_capability: 1,
                minimum_receiver_families_per_capability: 1,
                minimum_seeds_per_capability: 1,
                require_unseen_receiver: true,
                minimum_target_score: 0.9,
                minimum_preservation_score: 0.9,
                minimum_identity_margin: 0.5,
                minimum_success_probability: 0.0,
                confidence_z: 1.96,
            },
        };
        let universality_receipt = universality_input.execute().unwrap();
        let receipt = evaluate_universal_promotion_gate(&UniversalPromotionGateRequest {
            schema: "cerebro.tidex.universal_promotion_gate_request/v1".into(),
            selection_input,
            selection_receipt,
            universality_input,
            universality_receipt,
            shadow_evaluations: vec![],
            policy: UniversalPromotionPolicy {
                schema: "cerebro.tidex.universal_promotion_policy/v1".into(),
                minimum_universality_n: 1,
                minimum_global_wilson_lower_bound: 0.0,
                require_all_selected_candidates_evaluated: true,
            },
        })
        .unwrap();
        assert_eq!(receipt.readiness, UniversalPromotionReadiness::Rejected);
        assert!(receipt
            .blockers
            .contains(&UniversalPromotionBlocker::SelectedCandidateMissingEvaluation));
        assert!(!receipt.authorizes_activation);
    }
}
