//! Verified numerical core for the shadow LowRank/LoRA backend.
//!
//! This module factors an already compiled dense tensor delta. It does not
//! define capability identity and has no model-loading or activation API.

use crate::error::{BrainError, BrainResult};
use crate::linalg::{norm, symmetric_eigen_jacobi, Matrix};
use crate::solver_portfolio::CandidateRepresentation;
use serde::{Deserialize, Serialize};

const MAX_FACTOR_INPUT_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_FACTOR_IDENTITY_ELEMENTS: usize = 16 * 1024 * 1024;
const MAX_SHADOW_RANK: usize = 64;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct LowRankShadowPolicy {
    pub schema: String,
    pub maximum_rank: usize,
    pub relative_reconstruction_tolerance: f64,
    pub absolute_reconstruction_tolerance: f64,
    pub minimum_parameter_reduction_ratio: f64,
    pub maximum_svd_sweeps: usize,
}

impl LowRankShadowPolicy {
    pub fn validate(&self) -> BrainResult<()> {
        if self.schema != "cerebro.tidex.low_rank_shadow_policy/v1"
            || self.maximum_rank == 0
            || self.maximum_rank > MAX_SHADOW_RANK
            || !self.relative_reconstruction_tolerance.is_finite()
            || self.relative_reconstruction_tolerance < 0.0
            || !self.absolute_reconstruction_tolerance.is_finite()
            || self.absolute_reconstruction_tolerance < 0.0
            || self.relative_reconstruction_tolerance == 0.0
                && self.absolute_reconstruction_tolerance == 0.0
            || !self.minimum_parameter_reduction_ratio.is_finite()
            || !(0.0..1.0).contains(&self.minimum_parameter_reduction_ratio)
            || self.maximum_svd_sweeps == 0
        {
            return Err(BrainError::Invalid("low_rank_shadow_policy_invalid".into()));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct VerifiedLowRankFactors {
    pub schema: String,
    pub rows: usize,
    pub columns: usize,
    pub rank: usize,
    pub left: Vec<f64>,
    pub right: Vec<f64>,
    pub absolute_reconstruction_error: f64,
    pub relative_reconstruction_error: f64,
    pub dense_parameter_count: usize,
    pub factor_parameter_count: usize,
    pub parameter_reduction_ratio: f64,
}

impl VerifiedLowRankFactors {
    pub fn materialize_dense(&self) -> BrainResult<Vec<f64>> {
        CandidateRepresentation::LowRank {
            rows: self.rows,
            columns: self.columns,
            rank: self.rank,
            left: self.left.clone(),
            right: self.right.clone(),
        }
        .materialize_dense()
    }
}

pub fn factor_dense_delta_verified(
    rows: usize,
    columns: usize,
    dense: &[f64],
    policy: &LowRankShadowPolicy,
) -> BrainResult<VerifiedLowRankFactors> {
    policy.validate()?;
    let dense_count = rows
        .checked_mul(columns)
        .ok_or_else(|| BrainError::Invalid("low_rank_dense_shape_overflow".into()))?;
    let identity_count = columns
        .checked_mul(columns)
        .ok_or_else(|| BrainError::Invalid("low_rank_identity_shape_overflow".into()))?;
    if rows == 0
        || columns == 0
        || dense.len() != dense_count
        || dense_count > MAX_FACTOR_INPUT_ELEMENTS
        || identity_count > MAX_FACTOR_IDENTITY_ELEMENTS
        || dense.iter().any(|value| !value.is_finite())
    {
        return Err(BrainError::Invalid("low_rank_dense_input_invalid".into()));
    }
    let target_norm = norm(dense)?;
    let scale = dense
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f64, f64::max);
    if scale == 0.0 {
        return Err(BrainError::Numerical(
            "low_rank_zero_delta_has_no_factors".into(),
        ));
    }
    let spectral_dimension = rows.min(columns);
    let mut gram = Matrix::zeros(spectral_dimension, spectral_dimension);
    if rows <= columns {
        for first in 0..rows {
            for second in first..rows {
                let value = (0..columns)
                    .map(|column| {
                        (dense[first * columns + column] / scale)
                            * (dense[second * columns + column] / scale)
                    })
                    .sum::<f64>();
                gram.set(first, second, value);
                gram.set(second, first, value);
            }
        }
    } else {
        for first in 0..columns {
            for second in first..columns {
                let value = (0..rows)
                    .map(|row| {
                        (dense[row * columns + first] / scale)
                            * (dense[row * columns + second] / scale)
                    })
                    .sum::<f64>();
                gram.set(first, second, value);
                gram.set(second, first, value);
            }
        }
    }
    let rotations = spectral_dimension
        .checked_mul(spectral_dimension)
        .and_then(|value| value.checked_mul(policy.maximum_svd_sweeps))
        .ok_or_else(|| BrainError::Invalid("low_rank_svd_work_overflow".into()))?;
    let eigen = symmetric_eigen_jacobi(&gram, 1.0e-12, rotations)?;
    let maximum_rank = policy.maximum_rank.min(eigen.len());
    for rank in 1..=maximum_rank {
        let factor_count = rows
            .checked_mul(rank)
            .and_then(|value| value.checked_add(rank.checked_mul(columns)?))
            .ok_or_else(|| BrainError::Invalid("low_rank_factor_shape_overflow".into()))?;
        let parameter_reduction_ratio = 1.0 - factor_count as f64 / dense_count as f64;
        if parameter_reduction_ratio < policy.minimum_parameter_reduction_ratio {
            continue;
        }
        let mut left = vec![0.0; rows * rank];
        let mut right = vec![0.0; rank * columns];
        for component in 0..rank {
            let vector = &eigen[component].1;
            let scaled_sigma = eigen[component].0.sqrt();
            let balanced_scale = scale.sqrt() * scaled_sigma.sqrt();
            if !scaled_sigma.is_finite()
                || scaled_sigma <= 0.0
                || !balanced_scale.is_finite()
                || balanced_scale <= 0.0
            {
                return Err(BrainError::Numerical(
                    "low_rank_singular_value_invalid".into(),
                ));
            }
            if rows <= columns {
                for row in 0..rows {
                    left[row * rank + component] = vector[row] * balanced_scale;
                }
                for column in 0..columns {
                    right[component * columns + column] = (0..rows)
                        .map(|row| {
                            vector[row] * (dense[row * columns + column] / scale) / scaled_sigma
                        })
                        .sum::<f64>()
                        * balanced_scale;
                }
            } else {
                for row in 0..rows {
                    left[row * rank + component] = (0..columns)
                        .map(|column| {
                            (dense[row * columns + column] / scale) * vector[column] / scaled_sigma
                        })
                        .sum::<f64>()
                        * balanced_scale;
                }
                for column in 0..columns {
                    right[component * columns + column] = vector[column] * balanced_scale;
                }
            }
        }
        let candidate = CandidateRepresentation::LowRank {
            rows,
            columns,
            rank,
            left: left.clone(),
            right: right.clone(),
        };
        let reconstructed = candidate.materialize_dense()?;
        let residual = dense
            .iter()
            .zip(&reconstructed)
            .map(|(expected, actual)| expected - actual)
            .collect::<Vec<_>>();
        let absolute = norm(&residual)?;
        let relative = absolute / target_norm;
        if absolute.is_finite()
            && relative.is_finite()
            && (absolute <= policy.absolute_reconstruction_tolerance
                || relative <= policy.relative_reconstruction_tolerance)
        {
            return Ok(VerifiedLowRankFactors {
                schema: "cerebro.tidex.verified_low_rank_factors/v1".into(),
                rows,
                columns,
                rank,
                left,
                right,
                absolute_reconstruction_error: absolute,
                relative_reconstruction_error: relative,
                dense_parameter_count: dense_count,
                factor_parameter_count: factor_count,
                parameter_reduction_ratio,
            });
        }
    }
    Err(BrainError::Numerical(
        "low_rank_factorization_not_admissible".into(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn policy(maximum_rank: usize) -> LowRankShadowPolicy {
        LowRankShadowPolicy {
            schema: "cerebro.tidex.low_rank_shadow_policy/v1".into(),
            maximum_rank,
            relative_reconstruction_tolerance: 1.0e-10,
            absolute_reconstruction_tolerance: 1.0e-10,
            minimum_parameter_reduction_ratio: 0.4,
            maximum_svd_sweeps: 100,
        }
    }

    #[test]
    fn exact_low_rank_delta_is_factored_and_full_rank_delta_is_rejected() {
        let left = [1.0, 2.0, 3.0, 4.0];
        let right = [0.5, -1.0, 2.0, 0.25];
        let dense = left
            .iter()
            .flat_map(|left| right.iter().map(move |right| left * right))
            .collect::<Vec<_>>();
        let factors = factor_dense_delta_verified(4, 4, &dense, &policy(1)).unwrap();
        assert_eq!(factors.rank, 1);
        assert!(factors.parameter_reduction_ratio >= 0.4);
        let reconstructed = factors.materialize_dense().unwrap();
        assert!(dense
            .iter()
            .zip(reconstructed)
            .all(|(expected, actual)| (expected - actual).abs() < 1.0e-10));

        let identity = (0..16)
            .map(|index| f64::from(index / 4 == index % 4))
            .collect::<Vec<_>>();
        assert!(factor_dense_delta_verified(4, 4, &identity, &policy(1)).is_err());
    }

    #[test]
    fn rectangular_orientations_and_finite_scaling_reconstruct() {
        for (rows, columns, scale) in [(3, 8, 1.0e100), (8, 3, 1.0e-100)] {
            let left = (0..rows)
                .map(|index| (index + 1) as f64 * scale)
                .collect::<Vec<_>>();
            let right = (0..columns)
                .map(|index| (index as f64 + 0.5) / scale.sqrt())
                .collect::<Vec<_>>();
            let dense = (0..rows)
                .flat_map(|row| {
                    let left = &left;
                    let right = &right;
                    (0..columns).map(move |column| left[row] * right[column])
                })
                .collect::<Vec<_>>();
            let factors = factor_dense_delta_verified(rows, columns, &dense, &policy(1)).unwrap();
            assert_eq!(factors.rank, 1);
            assert!(factors.relative_reconstruction_error < 1.0e-10);
            assert!(factors
                .materialize_dense()
                .unwrap()
                .iter()
                .all(|value| value.is_finite()));
        }
    }
}
