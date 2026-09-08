//! Laboratory-root admission. Experimental state must never share a root with production.

use crate::error::{BrainError, BrainResult};
use crate::security::verify_private_root;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LabRoots {
    pub lab_root: PathBuf,
    pub state_root: PathBuf,
    pub artifact_root: PathBuf,
}

impl LabRoots {
    pub fn open(
        lab_root: &Path,
        state_root: &Path,
        artifact_root: &Path,
        production_root: &Path,
    ) -> BrainResult<Self> {
        let lab_root = verify_private_root(lab_root)?;
        let state_root = state_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("laboratory_state_root_unavailable".into()))?;
        let artifact_root = artifact_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("laboratory_artifact_root_unavailable".into()))?;
        let production_root = production_root
            .canonicalize()
            .map_err(|_| BrainError::Invalid("production_root_unavailable".into()))?;
        if !state_root.starts_with(&lab_root)
            || !artifact_root.starts_with(&lab_root)
            || state_root == artifact_root
            || state_root.starts_with(&production_root)
            || artifact_root.starts_with(&production_root)
        {
            return Err(BrainError::Integrity(
                "laboratory_roots_not_isolated".into(),
            ));
        }
        Ok(Self {
            lab_root,
            state_root,
            artifact_root,
        })
    }
}
