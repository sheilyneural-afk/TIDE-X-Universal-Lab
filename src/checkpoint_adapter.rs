//! Read-only, content-authenticated adapter for real SafeTensors checkpoints.
//!
//! It inspects headers and exact file bytes without loading tensor payloads.
//! Unsupported or ambiguous encodings are rejected rather than guessed.

use crate::architecture_families::{fingerprint_architecture, ArchitectureFamilyFingerprint};
use crate::block_tomography::{BlockShapeSpec, ParameterBlockLayout, ParameterLayoutArtifact};
use crate::digest::{sha256_file, Sha256Digest};
use crate::error::{BrainError, BrainResult};
use crate::identity::{ArchitectureId, ModelId, TensorId};
use crate::receiver_layout::{
    FloatingScalarType, ReceiverMaterializationLayout, ReceiverScalarEncoding,
    ReceiverTensorPartitioning, ReceiverTensorPhysicalSpec,
};
use crate::receiver_profile::{
    CapabilityModality, MaterializationStrategy, ReceiverArchitecture, ReceiverProfile,
    ReceiverRegion,
};
use crate::receiver_profiler::ReceiverSnapshotBinding;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

const MAX_CHECKPOINT_FILES: usize = 16_384;
const MAX_HEADER_BYTES: u64 = 64 * 1024 * 1024;
const MAX_AUXILIARY_BYTES: u64 = 256 * 1024 * 1024;
const MAX_CHECKPOINT_BYTES: u64 = 1 << 50;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SafeTensorsReceiverRequest {
    pub schema: String,
    pub model_id: ModelId,
    pub architecture_id: ArchitectureId,
    pub architecture: ReceiverArchitecture,
    pub modalities: BTreeSet<CapabilityModality>,
    pub supports_persistent_state: bool,
    pub checkpoint_files: Vec<PathBuf>,
    pub configuration_file: PathBuf,
    pub tokenizer_file: PathBuf,
    pub supported_strategies: BTreeSet<MaterializationStrategy>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct InspectedReceiverArtifacts {
    pub schema: String,
    pub profile: ReceiverProfile,
    pub layout: ReceiverMaterializationLayout,
    pub snapshot: ReceiverSnapshotBinding,
    pub architecture_fingerprint: ArchitectureFamilyFingerprint,
    pub checkpoint_file_sha256: BTreeMap<PathBuf, Sha256Digest>,
    pub manifest_sha256: Sha256Digest,
}

#[derive(Deserialize)]
struct TensorHeader {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

fn confined(root: &Path, relative: &Path) -> BrainResult<PathBuf> {
    if relative.is_absolute()
        || relative
            .components()
            .any(|part| matches!(part, std::path::Component::ParentDir))
    {
        return Err(BrainError::Invalid("checkpoint_path_not_relative".into()));
    }
    let canonical_root = root.canonicalize()?;
    let path = canonical_root.join(relative).canonicalize()?;
    if !path.starts_with(&canonical_root) || !path.metadata()?.is_file() {
        return Err(BrainError::Invalid(
            "checkpoint_path_not_confined_file".into(),
        ));
    }
    Ok(path)
}

fn digest_file(path: &Path, maximum: u64) -> BrainResult<Sha256Digest> {
    let metadata = path.metadata()?;
    if metadata.len() > maximum {
        return Err(BrainError::Invalid("checkpoint_file_too_large".into()));
    }
    sha256_file(path)
}

fn digest_file_exact(path: &Path, maximum: u64) -> BrainResult<Sha256Digest> {
    let metadata = path.metadata()?;
    if metadata.len() > maximum {
        return Err(BrainError::Invalid("checkpoint_file_too_large".into()));
    }
    let mut file = File::open(path)?;
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    file.read_to_end(&mut bytes)?;
    Ok(Sha256Digest::digest_bytes(&bytes))
}

fn encoding(dtype: &str) -> BrainResult<(ReceiverScalarEncoding, u64)> {
    match dtype {
        "F64" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float64,
            },
            8,
        )),
        "F32" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float32,
            },
            4,
        )),
        "BF16" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Bfloat16,
            },
            2,
        )),
        "F16" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float16,
            },
            2,
        )),
        "F8_E4M3" | "F8_E4M3FN" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float8E4m3,
            },
            1,
        )),
        "F8_E5M2" => Ok((
            ReceiverScalarEncoding::Floating {
                scalar_type: FloatingScalarType::Float8E5m2,
            },
            1,
        )),
        _ => Err(BrainError::Invalid(format!(
            "checkpoint_dtype_requires_explicit_quantization_contract:{dtype}"
        ))),
    }
}

fn read_headers(
    path: &Path,
) -> BrainResult<BTreeMap<String, (TensorHeader, ReceiverScalarEncoding)>> {
    let mut file = File::open(path)?;
    let file_size = file.metadata()?.len();
    let mut length = [0u8; 8];
    file.read_exact(&mut length)?;
    let header_len = u64::from_le_bytes(length);
    if header_len == 0 || header_len > MAX_HEADER_BYTES || header_len > file_size.saturating_sub(8)
    {
        return Err(BrainError::Invalid(
            "safetensors_header_length_invalid".into(),
        ));
    }
    let mut bytes = vec![
        0u8;
        usize::try_from(header_len).map_err(|_| BrainError::Invalid(
            "safetensors_header_overflow".into()
        ))?
    ];
    file.read_exact(&mut bytes)?;
    let value: serde_json::Value = serde_json::from_slice(&bytes)?;
    let object = value
        .as_object()
        .ok_or_else(|| BrainError::Invalid("safetensors_header_not_object".into()))?;
    let data_bytes = file_size - 8 - header_len;
    let mut tensors = BTreeMap::new();
    let mut ranges = Vec::new();
    for (name, value) in object {
        if name == "__metadata__" {
            continue;
        }
        let header: TensorHeader = serde_json::from_value(value.clone())?;
        let (encoding, bytes_per_element) = encoding(&header.dtype)?;
        let count = header.shape.iter().try_fold(1u64, |a, v| {
            a.checked_mul(*v as u64)
                .ok_or_else(|| BrainError::Invalid("checkpoint_tensor_shape_overflow".into()))
        })?;
        if name.trim().is_empty()
            || name != name.trim()
            || header.shape.is_empty()
            || header.shape.contains(&0)
            || header.data_offsets[0] > header.data_offsets[1]
            || header.data_offsets[1] > data_bytes
            || header.data_offsets[1] - header.data_offsets[0]
                != count
                    .checked_mul(bytes_per_element)
                    .ok_or_else(|| BrainError::Invalid("checkpoint_tensor_bytes_overflow".into()))?
        {
            return Err(BrainError::Invalid(
                "safetensors_tensor_header_invalid".into(),
            ));
        }
        ranges.push(header.data_offsets);
        tensors.insert(name.clone(), (header, encoding));
    }
    if tensors.is_empty() {
        return Err(BrainError::Invalid("safetensors_checkpoint_empty".into()));
    }
    ranges.sort_by_key(|range| range[0]);
    let mut expected_start = 0u64;
    for [start, end] in ranges {
        if start != expected_start {
            return Err(BrainError::Invalid(
                "safetensors_data_offsets_not_contiguous".into(),
            ));
        }
        expected_start = end;
    }
    if expected_start != data_bytes {
        return Err(BrainError::Invalid(
            "safetensors_data_region_not_fully_described".into(),
        ));
    }
    file.seek(SeekFrom::End(0))?;
    Ok(tensors)
}

pub fn inspect_safetensors_receiver(
    root: &Path,
    request: &SafeTensorsReceiverRequest,
) -> BrainResult<InspectedReceiverArtifacts> {
    if request.schema != "cerebro.tidex.safetensors_receiver_request/v1"
        || request.checkpoint_files.is_empty()
        || request.checkpoint_files.len() > MAX_CHECKPOINT_FILES
        || request.modalities.is_empty()
        || request.supported_strategies.is_empty()
    {
        return Err(BrainError::Invalid(
            "safetensors_receiver_request_invalid".into(),
        ));
    }
    let mut files = request.checkpoint_files.clone();
    files.sort();
    if files.windows(2).any(|p| p[0] >= p[1]) {
        return Err(BrainError::Invalid("checkpoint_files_not_unique".into()));
    }
    let mut all = BTreeMap::new();
    let mut file_digests = BTreeMap::new();
    let mut total_bytes = 0u64;
    for relative in files {
        let path = confined(root, &relative)?;
        total_bytes = total_bytes
            .checked_add(path.metadata()?.len())
            .ok_or_else(|| BrainError::Invalid("checkpoint_total_overflow".into()))?;
        if total_bytes > MAX_CHECKPOINT_BYTES {
            return Err(BrainError::Invalid("checkpoint_total_limit".into()));
        }
        let headers = read_headers(&path)?;
        for (name, spec) in headers {
            if all.insert(name, spec).is_some() {
                return Err(BrainError::Invalid("checkpoint_duplicate_tensor".into()));
            }
        }
        file_digests.insert(relative, digest_file(&path, MAX_CHECKPOINT_BYTES)?);
    }
    let config_path = confined(root, &request.configuration_file)?;
    if config_path.metadata()?.len() > MAX_AUXILIARY_BYTES {
        return Err(BrainError::Invalid("checkpoint_file_too_large".into()));
    }
    let config_bytes = std::fs::read(&config_path)?;
    let config = Sha256Digest::digest_bytes(&config_bytes);
    let tokenizer = digest_file_exact(
        &confined(root, &request.tokenizer_file)?,
        MAX_AUXILIARY_BYTES,
    )?;
    let tensor_names = all.keys().cloned().collect::<Vec<_>>();
    let architecture_fingerprint = fingerprint_architecture(&config_bytes, &tensor_names)?;
    if request.architecture != ReceiverArchitecture::Unknown
        && architecture_fingerprint.receiver_architecture != ReceiverArchitecture::Unknown
        && request.architecture != architecture_fingerprint.receiver_architecture
    {
        return Err(BrainError::Integrity(
            "declared_receiver_architecture_conflicts_with_checkpoint".into(),
        ));
    }
    let resolved_architecture = if request.architecture == ReceiverArchitecture::Unknown {
        architecture_fingerprint.receiver_architecture
    } else {
        request.architecture
    };
    let mut shapes = Vec::with_capacity(all.len());
    let mut regions = Vec::with_capacity(all.len());
    let mut physical = Vec::with_capacity(all.len());
    for (name, (header, encoding)) in all {
        let tensor_id = TensorId::parse(&name)?;
        let count = header.shape.iter().try_fold(1usize, |a, v| {
            a.checked_mul(*v)
                .ok_or_else(|| BrainError::Invalid("checkpoint_parameter_overflow".into()))
        })?;
        shapes.push(BlockShapeSpec {
            name: name.clone(),
            shape: header.shape,
            count,
        });
        regions.push(ReceiverRegion {
            tensor_id: tensor_id.clone(),
            parameter_count: u64::try_from(count)
                .map_err(|_| BrainError::Invalid("checkpoint_parameter_overflow".into()))?,
            supported_strategies: request.supported_strategies.clone(),
        });
        physical.push(ReceiverTensorPhysicalSpec {
            tensor_id,
            encoding,
            partitioning: ReceiverTensorPartitioning::Replicated,
        });
    }
    let geometry = ParameterLayoutArtifact::new(ParameterBlockLayout::from_shapes(&shapes)?)?;
    let profile = ReceiverProfile {
        schema: "cerebro.tidex.receiver_profile/v1".into(),
        model_id: request.model_id.clone(),
        architecture_id: request.architecture_id.clone(),
        architecture: resolved_architecture,
        modalities: request.modalities.clone(),
        supports_persistent_state: request.supports_persistent_state,
        parameter_dimension: geometry.total_parameter_count,
        regions,
    };
    let layout = ReceiverMaterializationLayout::create(&profile, geometry, physical, vec![])?;
    let snapshot_digest = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:SAFETENSORS-SNAPSHOT:v1\0",
        &serde_json::to_vec(&file_digests)?,
    );
    let snapshot = ReceiverSnapshotBinding::create(
        &profile,
        snapshot_digest,
        config,
        tokenizer,
        layout.manifest_sha256.clone(),
    )?;
    let mut result = InspectedReceiverArtifacts {
        schema: "cerebro.tidex.inspected_receiver_artifacts/v1".into(),
        profile,
        layout,
        snapshot,
        architecture_fingerprint,
        checkpoint_file_sha256: file_digests,
        manifest_sha256: Sha256Digest::zero(),
    };
    let mut unsigned = result.clone();
    unsigned.manifest_sha256 = Sha256Digest::zero();
    result.manifest_sha256 = Sha256Digest::digest_domain(
        b"CEREBRO:TIDEX:INSPECTED-RECEIVER-ARTIFACTS:v1\0",
        &serde_json::to_vec(&unsigned)?,
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn inspects_exact_safetensors_geometry_and_rejects_ambiguous_dtype() {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root =
            std::env::temp_dir().join(format!("tidex-safetensors-{}-{nonce}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        fs::write(root.join("config.json"), b"{}").unwrap();
        fs::write(root.join("tokenizer.json"), b"{}").unwrap();
        let header = br#"{"layer.weight":{"dtype":"F32","shape":[2,2],"data_offsets":[0,16]}}"#;
        let mut model = File::create(root.join("model.safetensors")).unwrap();
        model
            .write_all(&(header.len() as u64).to_le_bytes())
            .unwrap();
        model.write_all(header).unwrap();
        model.write_all(&[0u8; 16]).unwrap();
        let request = SafeTensorsReceiverRequest {
            schema: "cerebro.tidex.safetensors_receiver_request/v1".into(),
            model_id: ModelId::parse("receiver.fixture").unwrap(),
            architecture_id: ArchitectureId::parse("transformer.fixture").unwrap(),
            architecture: ReceiverArchitecture::Transformer,
            modalities: BTreeSet::from([CapabilityModality::Text]),
            supports_persistent_state: false,
            checkpoint_files: vec!["model.safetensors".into()],
            configuration_file: "config.json".into(),
            tokenizer_file: "tokenizer.json".into(),
            supported_strategies: BTreeSet::from([MaterializationStrategy::DenseDelta]),
        };
        let inspected = inspect_safetensors_receiver(&root, &request).unwrap();
        assert_eq!(inspected.profile.parameter_dimension, 4);
        assert_eq!(inspected.layout.geometry.layout.blocks[0].shape, vec![2, 2]);
        assert_ne!(
            inspected.snapshot.model_snapshot_sha256,
            Sha256Digest::zero()
        );
        fs::remove_dir_all(root).unwrap();
    }
}
