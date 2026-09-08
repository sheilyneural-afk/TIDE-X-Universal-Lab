//! Evidence-driven backend selection and comparative admission gates.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::receiver_profile::MaterializationStrategy;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;

const MAX_EVALUATIONS: usize = 1_024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum ComparativeControl {
    UnmodifiedReceiver,
    DenseDelta,
    ConventionalLowRank,
    WrongCapabilityIr,
    RandomDelta,
    MeanCapability,
    NearestCapability,
    AlternativeBackend,
    NonTargetPreservation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackendEvaluation {
    pub schema: String,
    pub candidate_sha256: Sha256Digest,
    pub strategy: MaterializationStrategy,
    pub functional_score: f64,
    pub functional_ci_lower: f64,
    pub preservation_score: f64,
    pub identity_margin: f64,
    pub numerical_stability: f64,
    pub normalized_risk: f64,
    pub latency_micros: u64,
    pub resident_bytes: u64,
    pub completed_controls: BTreeSet<ComparativeControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackendSelectionPolicy {
    pub schema: String,
    pub minimum_functional_ci_lower: f64,
    pub minimum_preservation_score: f64,
    pub minimum_identity_margin: f64,
    pub minimum_numerical_stability: f64,
    pub maximum_normalized_risk: f64,
    pub maximum_latency_micros: u64,
    pub maximum_resident_bytes: u64,
    pub functional_weight: f64,
    pub preservation_weight: f64,
    pub stability_weight: f64,
    pub risk_weight: f64,
    pub latency_weight: f64,
    pub memory_weight: f64,
    pub required_controls: BTreeSet<ComparativeControl>,
    pub allow_hybrid: bool,
    pub minimum_hybrid_complementarity: f64,
}

impl BackendSelectionPolicy {
    pub fn rigorous_default(maximum_latency_micros: u64, maximum_resident_bytes: u64) -> Self {
        Self {
            schema: "cerebro.tidex.backend_selection_policy/v1".into(),
            minimum_functional_ci_lower: 0.8,
            minimum_preservation_score: 0.95,
            minimum_identity_margin: 0.05,
            minimum_numerical_stability: 0.99,
            maximum_normalized_risk: 0.1,
            maximum_latency_micros,
            maximum_resident_bytes,
            functional_weight: 0.35,
            preservation_weight: 0.25,
            stability_weight: 0.15,
            risk_weight: 0.1,
            latency_weight: 0.075,
            memory_weight: 0.075,
            required_controls: BTreeSet::from([
                ComparativeControl::UnmodifiedReceiver,
                ComparativeControl::DenseDelta,
                ComparativeControl::ConventionalLowRank,
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

    pub fn validate(&self) -> BrainResult<()> {
        let bounded = [
            self.minimum_functional_ci_lower,
            self.minimum_preservation_score,
            self.minimum_numerical_stability,
            self.maximum_normalized_risk,
            self.minimum_hybrid_complementarity,
        ];
        let weights = [
            self.functional_weight,
            self.preservation_weight,
            self.stability_weight,
            self.risk_weight,
            self.latency_weight,
            self.memory_weight,
        ];
        if self.schema != "cerebro.tidex.backend_selection_policy/v1"
            || bounded
                .iter()
                .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
            || !self.minimum_identity_margin.is_finite()
            || self.minimum_identity_margin < 0.0
            || self.maximum_latency_micros == 0
            || self.maximum_resident_bytes == 0
            || weights.iter().any(|v| !v.is_finite() || *v < 0.0)
            || weights.iter().sum::<f64>() <= 0.0
            || self.required_controls.is_empty()
        {
            return Err(BrainError::Invalid(
                "backend_selection_policy_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PairwiseComplementarity {
    pub first_candidate_sha256: Sha256Digest,
    pub second_candidate_sha256: Sha256Digest,
    pub held_out_gain: f64,
    pub preservation_delta: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct RankedBackend {
    pub candidate_sha256: Sha256Digest,
    pub strategy: MaterializationStrategy,
    pub utility: f64,
    pub pareto_optimal: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackendSelectionReceipt {
    pub schema: String,
    pub policy_sha256: Sha256Digest,
    pub evidence_sha256: Sha256Digest,
    pub ranked: Vec<RankedBackend>,
    pub selected_strategy: MaterializationStrategy,
    pub selected_candidates: Vec<Sha256Digest>,
    pub manifest_sha256: Sha256Digest,
}

impl BackendSelectionReceipt {
    pub fn validate_against(&self, input: &BackendSelectionInput) -> BrainResult<()> {
        if self != &input.execute()? {
            return Err(BrainError::Integrity(
                "backend_selection_receipt_invalid".into(),
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct BackendSelectionInput {
    pub schema: String,
    pub evaluations: Vec<BackendEvaluation>,
    pub complementarity: Vec<PairwiseComplementarity>,
    pub policy: BackendSelectionPolicy,
}

impl BackendSelectionInput {
    pub fn execute(&self) -> BrainResult<BackendSelectionReceipt> {
        if self.schema != "cerebro.tidex.backend_selection_input/v1" {
            return Err(BrainError::Invalid(
                "backend_selection_input_invalid".into(),
            ));
        }
        select_materialization_backend(&self.evaluations, &self.complementarity, &self.policy)
    }
}

fn validate_evaluation(e: &BackendEvaluation) -> BrainResult<()> {
    let unit = [
        e.functional_score,
        e.functional_ci_lower,
        e.preservation_score,
        e.numerical_stability,
        e.normalized_risk,
    ];
    if e.schema != "cerebro.tidex.backend_evaluation/v1"
        || e.candidate_sha256 == Sha256Digest::zero()
        || unit
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        || !e.identity_margin.is_finite()
        || e.identity_margin < 0.0
    {
        return Err(BrainError::Invalid("backend_evaluation_invalid".into()));
    }
    Ok(())
}

fn dominates(a: &BackendEvaluation, b: &BackendEvaluation) -> bool {
    let no_worse = a.functional_ci_lower >= b.functional_ci_lower
        && a.preservation_score >= b.preservation_score
        && a.numerical_stability >= b.numerical_stability
        && a.normalized_risk <= b.normalized_risk
        && a.latency_micros <= b.latency_micros
        && a.resident_bytes <= b.resident_bytes;
    let better = a.functional_ci_lower > b.functional_ci_lower
        || a.preservation_score > b.preservation_score
        || a.numerical_stability > b.numerical_stability
        || a.normalized_risk < b.normalized_risk
        || a.latency_micros < b.latency_micros
        || a.resident_bytes < b.resident_bytes;
    no_worse && better
}

pub fn select_materialization_backend(
    evaluations: &[BackendEvaluation],
    complementarity: &[PairwiseComplementarity],
    policy: &BackendSelectionPolicy,
) -> BrainResult<BackendSelectionReceipt> {
    policy.validate()?;
    if evaluations.is_empty() || evaluations.len() > MAX_EVALUATIONS {
        return Err(BrainError::Invalid(
            "backend_evaluation_cardinality_invalid".into(),
        ));
    }
    let mut ids = BTreeSet::new();
    for e in evaluations {
        validate_evaluation(e)?;
        if !ids.insert(&e.candidate_sha256) {
            return Err(BrainError::Invalid("backend_evaluation_duplicate".into()));
        }
    }
    let admitted = evaluations
        .iter()
        .filter(|e| {
            e.functional_ci_lower >= policy.minimum_functional_ci_lower
                && e.preservation_score >= policy.minimum_preservation_score
                && e.identity_margin >= policy.minimum_identity_margin
                && e.numerical_stability >= policy.minimum_numerical_stability
                && e.normalized_risk <= policy.maximum_normalized_risk
                && e.latency_micros <= policy.maximum_latency_micros
                && e.resident_bytes <= policy.maximum_resident_bytes
                && policy.required_controls.is_subset(&e.completed_controls)
        })
        .collect::<Vec<_>>();
    if admitted.is_empty() {
        return Err(BrainError::Integrity(
            "no_backend_passed_comparative_gates".into(),
        ));
    }
    let admitted_ids = admitted
        .iter()
        .map(|evaluation| &evaluation.candidate_sha256)
        .collect::<BTreeSet<_>>();
    let weight_sum = policy.functional_weight
        + policy.preservation_weight
        + policy.stability_weight
        + policy.risk_weight
        + policy.latency_weight
        + policy.memory_weight;
    let mut ranked = admitted
        .iter()
        .map(|e| {
            let utility = (policy.functional_weight * e.functional_ci_lower
                + policy.preservation_weight * e.preservation_score
                + policy.stability_weight * e.numerical_stability
                + policy.risk_weight * (1.0 - e.normalized_risk)
                + policy.latency_weight
                    * (1.0 - e.latency_micros as f64 / policy.maximum_latency_micros as f64)
                + policy.memory_weight
                    * (1.0 - e.resident_bytes as f64 / policy.maximum_resident_bytes as f64))
                / weight_sum;
            RankedBackend {
                candidate_sha256: e.candidate_sha256.clone(),
                strategy: e.strategy,
                utility,
                pareto_optimal: !admitted.iter().any(|other| {
                    other.candidate_sha256 != e.candidate_sha256 && dominates(other, e)
                }),
            }
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|a, b| {
        b.utility
            .partial_cmp(&a.utility)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.candidate_sha256.as_str().cmp(b.candidate_sha256.as_str()))
    });
    let mut selected_strategy = ranked[0].strategy;
    let mut selected_candidates = vec![ranked[0].candidate_sha256.clone()];
    if policy.allow_hybrid {
        let best_pair =
            complementarity
                .iter()
                .filter(|pair| {
                    pair.held_out_gain.is_finite()
                        && pair.preservation_delta.is_finite()
                        && (0.0..=1.0).contains(&pair.held_out_gain)
                        && (-1.0..=1.0).contains(&pair.preservation_delta)
                        && pair.held_out_gain >= policy.minimum_hybrid_complementarity
                        && pair.preservation_delta >= 0.0
                        && admitted_ids.contains(&pair.first_candidate_sha256)
                        && admitted_ids.contains(&pair.second_candidate_sha256)
                        && pair.first_candidate_sha256 != pair.second_candidate_sha256
                        && evaluations
                            .iter()
                            .find(|value| value.candidate_sha256 == pair.first_candidate_sha256)
                            .zip(evaluations.iter().find(|value| {
                                value.candidate_sha256 == pair.second_candidate_sha256
                            }))
                            .is_some_and(|(first, second)| first.strategy != second.strategy)
                })
                .max_by(|a, b| {
                    a.held_out_gain
                        .partial_cmp(&b.held_out_gain)
                        .unwrap_or(Ordering::Equal)
                });
        if let Some(pair) = best_pair {
            selected_strategy = MaterializationStrategy::Hybrid;
            selected_candidates = vec![
                pair.first_candidate_sha256.clone(),
                pair.second_candidate_sha256.clone(),
            ];
            selected_candidates.sort_by(|a, b| a.as_str().cmp(b.as_str()));
        }
    }
    let policy_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:BACKEND-SELECTION-POLICY:v1\0",
        &serde_json::to_vec(policy)?,
    );
    let evidence_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:BACKEND-SELECTION-EVIDENCE:v1\0",
        &serde_json::to_vec(&(evaluations, complementarity))?,
    );
    let mut receipt = BackendSelectionReceipt {
        schema: "cerebro.tidex.backend_selection_receipt/v1".into(),
        policy_sha256,
        evidence_sha256,
        ranked,
        selected_strategy,
        selected_candidates,
        manifest_sha256: Sha256Digest::zero(),
    };
    let mut unsigned = receipt.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    receipt.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:BACKEND-SELECTION-RECEIPT:v1\0",
        &serde_json::to_vec(&unsigned)?,
    );
    Ok(receipt)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn evaluation(id: &[u8], strategy: MaterializationStrategy, score: f64) -> BackendEvaluation {
        BackendEvaluation {
            schema: "cerebro.tidex.backend_evaluation/v1".into(),
            candidate_sha256: Sha256Digest::digest_bytes(id),
            strategy,
            functional_score: score,
            functional_ci_lower: score,
            preservation_score: 0.99,
            identity_margin: 0.2,
            numerical_stability: 1.0,
            normalized_risk: 0.01,
            latency_micros: 10,
            resident_bytes: 10,
            completed_controls: BackendSelectionPolicy::rigorous_default(100, 100)
                .required_controls,
        }
    }
    #[test]
    fn selects_evidence_not_a_default_backend() {
        let policy = BackendSelectionPolicy::rigorous_default(100, 100);
        let receipt = select_materialization_backend(
            &[
                evaluation(b"lora", MaterializationStrategy::LowRank, 0.85),
                evaluation(b"sparse", MaterializationStrategy::SparseDelta, 0.95),
            ],
            &[],
            &policy,
        )
        .unwrap();
        assert_eq!(
            receipt.selected_strategy,
            MaterializationStrategy::SparseDelta
        );
    }
}
