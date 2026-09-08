//! Inert, replay-bound dense-delta materialization for laboratory receivers.
//!
//! This backend converts the compiler's authenticated flat `target_delta` into
//! exact tensor-shaped blocks. It cannot load a model, mutate parameters, or
//! activate the result.

use crate::authority::{write_or_verify_immutable, PrivateFileReference};
use crate::block_tomography::ParameterLayoutArtifact;
use crate::digest::{ParameterLayoutDigest, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::TensorId;
use crate::lab_isolation::LabRoots;
use crate::receiver_profile::MaterializationStrategy;
use crate::receiver_profiler::validate_receiver_parameter_layout;
use crate::universal_capability_compiler::{
    replay_universal_capability_shadow_plan, UniversalCapabilityPlanningRequest,
    UniversalCapabilityShadowPlanReceipt,
};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const MAX_DENSE_SHADOW_BYTES: u64 = 256 * 1024 * 1024;
const MAX_DENSE_SHADOW_ELEMENTS: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowDenseTensorDelta {
    pub tensor_id: TensorId,
    pub shape: Vec<usize>,
    pub values: Vec<f64>,
    pub values_sha256: Sha256Digest,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowDenseDeltaCandidate {
    pub schema: String,
    pub planning_request_sha256: Sha256Digest,
    pub parameter_layout_sha256: ParameterLayoutDigest,
    pub target_delta_sha256: Sha256Digest,
    pub tensors: Vec<ShadowDenseTensorDelta>,
    pub manifest_sha256: Sha256Digest,
}

fn values_digest(values: &[f64]) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-DENSE-VALUES:v1\0",
        &serde_json::to_vec(values)?,
    ))
}

impl ShadowDenseDeltaCandidate {
    fn calculate_digest(&self) -> BrainResult<Sha256Digest> {
        let mut unsigned = self.clone();
        unsigned.manifest_sha256 = Sha256Digest::zero();
        Ok(Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-DENSE-DELTA-CANDIDATE:v1\0",
            &serde_json::to_vec(&unsigned)?,
        ))
    }

    pub fn validate(
        &self,
        request: &UniversalCapabilityPlanningRequest,
        receipt: &UniversalCapabilityShadowPlanReceipt,
        layout: &ParameterLayoutArtifact,
    ) -> BrainResult<()> {
        let expected = build_candidate(request, receipt, layout)?;
        if self != &expected {
            return Err(BrainError::Integrity(
                "shadow_dense_delta_candidate_invalid".into(),
            ));
        }
        Ok(())
    }
}

fn build_candidate(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ParameterLayoutArtifact,
) -> BrainResult<ShadowDenseDeltaCandidate> {
    replay_universal_capability_shadow_plan(request, receipt)?;
    validate_receiver_parameter_layout(
        &request.receiver_profile,
        &request.receiver_snapshot,
        layout,
    )?;
    let shadow = &receipt.shadow_plan;
    if shadow.materialization_plan.strategy != MaterializationStrategy::DenseDelta {
        return Err(BrainError::Invalid(
            "shadow_dense_delta_strategy_required".into(),
        ));
    }
    let target = &shadow.compilation_receipt.compilation.receiver.target_delta;
    if target.len() > MAX_DENSE_SHADOW_ELEMENTS
        || target.iter().any(|value| !value.is_finite())
        || u64::try_from(target.len())
            .map_err(|_| BrainError::Invalid("shadow_dense_target_length_overflow".into()))?
            != layout.total_parameter_count
    {
        return Err(BrainError::Integrity(
            "shadow_dense_target_delta_invalid".into(),
        ));
    }
    let affected = shadow
        .materialization_plan
        .affected_regions
        .iter()
        .collect::<BTreeSet<_>>();
    let mut tensors = Vec::with_capacity(affected.len());
    for (block, region) in layout
        .layout
        .blocks
        .iter()
        .zip(&request.receiver_profile.regions)
    {
        let start = usize::try_from(block.offset)
            .map_err(|_| BrainError::Invalid("shadow_dense_offset_overflow".into()))?;
        let end = start
            .checked_add(block.count)
            .ok_or_else(|| BrainError::Invalid("shadow_dense_range_overflow".into()))?;
        let values = target
            .get(start..end)
            .ok_or_else(|| BrainError::Integrity("shadow_dense_range_invalid".into()))?;
        if affected.contains(&region.tensor_id) {
            tensors.push(ShadowDenseTensorDelta {
                tensor_id: region.tensor_id.clone(),
                shape: block.shape.clone(),
                values: values.to_vec(),
                values_sha256: values_digest(values)?,
            });
        } else if values.iter().any(|value| *value != 0.0) {
            return Err(BrainError::Integrity(
                "shadow_dense_nonzero_outside_planned_regions".into(),
            ));
        }
    }
    let mut candidate = ShadowDenseDeltaCandidate {
        schema: "cerebro.tidex.shadow_dense_delta_candidate/v1".into(),
        planning_request_sha256: receipt.planning_request_sha256.clone(),
        parameter_layout_sha256: layout.parameter_layout_sha256.clone(),
        target_delta_sha256: values_digest(target)?,
        tensors,
        manifest_sha256: Sha256Digest::zero(),
    };
    candidate.manifest_sha256 = candidate.calculate_digest()?;
    Ok(candidate)
}

pub fn materialize_replayed_dense_delta_shadow(
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ParameterLayoutArtifact,
) -> BrainResult<ShadowDenseDeltaCandidate> {
    let candidate = build_candidate(request, receipt, layout)?;
    candidate.validate(request, receipt, layout)?;
    Ok(candidate)
}

pub fn persist_dense_delta_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ParameterLayoutArtifact,
    candidate: &ShadowDenseDeltaCandidate,
) -> BrainResult<PrivateFileReference> {
    candidate.validate(request, receipt, layout)?;
    let bytes = serde_json::to_vec(candidate)?;
    if u64::try_from(bytes.len())
        .map_err(|_| BrainError::Invalid("shadow_dense_candidate_size_overflow".into()))?
        > MAX_DENSE_SHADOW_BYTES
    {
        return Err(BrainError::Invalid(
            "shadow_dense_candidate_too_large".into(),
        ));
    }
    let destination = roots
        .artifact_root()
        .join("dense-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    let sha256 = write_or_verify_immutable(roots.lab_root(), &destination, &bytes)?;
    Ok(PrivateFileReference::new(destination, sha256))
}

pub fn load_dense_delta_shadow(
    roots: &LabRoots,
    request: &UniversalCapabilityPlanningRequest,
    receipt: &UniversalCapabilityShadowPlanReceipt,
    layout: &ParameterLayoutArtifact,
    reference: &PrivateFileReference,
) -> BrainResult<ShadowDenseDeltaCandidate> {
    let bytes = reference.read_verified_bounded(roots.lab_root(), MAX_DENSE_SHADOW_BYTES)?;
    let candidate: ShadowDenseDeltaCandidate = serde_json::from_slice(&bytes)?;
    let expected_path = roots
        .artifact_root()
        .join("dense-shadow-candidates")
        .join(format!("{}.json", candidate.manifest_sha256));
    if reference.path != expected_path {
        return Err(BrainError::Integrity(
            "shadow_dense_candidate_path_invalid".into(),
        ));
    }
    candidate.validate(request, receipt, layout)?;
    Ok(candidate)
}
