//! Non-actuating candidate construction for receiver-coordinate experiments.
//!
//! This module intentionally has no filesystem, model-runtime, adapter, or
//! activation dependency.  It can only produce a digest-bound in-memory
//! candidate that a later, separately reviewed backend may consume.

use crate::digest::Sha256Digest;
use crate::error::{BrainError, BrainResult};
use crate::receiver_profile::{
    MaterializationPlan, MaterializationStrategy, PlanLifecycle, ReceiverProfile,
};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ShadowReceiverCoordinateCandidate {
    pub schema: String,
    pub plan_sha256: Sha256Digest,
    pub coordinate_sha256: Sha256Digest,
    pub receiver_parameter_dimension: u64,
    pub coordinates: Vec<f64>,
}

fn plan_digest(plan: &MaterializationPlan) -> BrainResult<Sha256Digest> {
    Ok(Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SHADOW-MATERIALIZATION-PLAN:v1\0",
        &serde_json::to_vec(plan)?,
    ))
}

/// Construct an inert candidate.  This does not serialize to a model format,
/// touch a tensor, or provide an activation method.
pub fn materialize_receiver_coordinates_shadow(
    profile: &ReceiverProfile,
    plan: &MaterializationPlan,
    coordinates: Vec<f64>,
) -> BrainResult<ShadowReceiverCoordinateCandidate> {
    profile.validate()?;
    if plan.schema != "cerebro.tidex.materialization_plan/v1"
        || plan.receiver_profile_sha256 != profile.digest()?
        || plan.lifecycle != PlanLifecycle::ShadowOnly
        || plan.strategy != MaterializationStrategy::ReceiverCoordinates
        || u64::try_from(coordinates.len())
            .map_err(|_| BrainError::Invalid("shadow_coordinate_length_overflow".into()))?
            != profile.parameter_dimension
        || coordinates.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid(
            "shadow_receiver_coordinate_candidate_invalid".into(),
        ));
    }
    let coordinate_bytes = serde_json::to_vec(&coordinates)?;
    Ok(ShadowReceiverCoordinateCandidate {
        schema: "cerebro.tidex.shadow_receiver_coordinate_candidate/v1".into(),
        plan_sha256: plan_digest(plan)?,
        coordinate_sha256: Sha256Digest::digest_domain(
            b"CEREBRO:TIDEX:SHADOW-RECEIVER-COORDINATES:v1\0",
            &coordinate_bytes,
        ),
        receiver_parameter_dimension: profile.parameter_dimension,
        coordinates,
    })
}
