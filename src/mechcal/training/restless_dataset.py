"""Load pooled restless-bandit transcripts without exposing audit covariates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AuditFields:
    condition_index: np.ndarray
    condition_name: np.ndarray
    kernel_sign: np.ndarray
    session_id: np.ndarray
    base_participant_id: np.ndarray


class RestlessTranscriptDataset(Dataset[tuple[torch.Tensor, torch.Tensor, int]]):
    """In-memory token dataset; audit fields are retained outside model examples."""

    def __init__(
        self,
        root: Path,
        split: str,
        condition: str,
        limit: int | None = None,
    ) -> None:
        files = sorted(root.glob(f"{split}_*.npz"))
        if not files:
            raise FileNotFoundError(f"no {split} shards found under {root}")
        model_arrays: dict[str, list[np.ndarray]] = {
            "tokens": [],
            "choice_target_mask": [],
        }
        audit_arrays: dict[str, list[np.ndarray]] = {
            "condition_index": [],
            "condition_name": [],
            "kernel_sign": [],
            "session_id": [],
            "base_participant_id": [],
        }
        for path in files:
            with np.load(path, allow_pickle=False) as shard:
                for key in model_arrays:
                    model_arrays[key].append(shard[key])
                for key in audit_arrays:
                    audit_arrays[key].append(shard[key])

        concatenated_model = {
            key: np.concatenate(value) for key, value in model_arrays.items()
        }
        concatenated_audit = {
            key: np.concatenate(value) for key, value in audit_arrays.items()
        }
        selected = np.flatnonzero(concatenated_audit["condition_name"] == condition)
        if not len(selected):
            available = sorted(set(concatenated_audit["condition_name"].tolist()))
            raise ValueError(f"unknown condition {condition!r}; available: {available}")
        # Sorting by the shared base ID makes a limit select the exact same
        # participants/schedules in every condition-specific model.
        selected = selected[
            np.argsort(concatenated_audit["base_participant_id"][selected])
        ]
        if limit is not None:
            selected = selected[:limit]

        self.tokens = torch.from_numpy(
            concatenated_model["tokens"][selected].astype(np.int64)
        )
        self.choice_target_mask = torch.from_numpy(
            concatenated_model["choice_target_mask"][selected].astype(bool)
        )
        self.audit = AuditFields(
            **{key: value[selected] for key, value in concatenated_audit.items()}
        )

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.tokens[index], self.choice_target_mask[index], index


def causal_batch(
    tokens: torch.Tensor, choice_target_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift a transcript so each target is predicted only from prior tokens."""
    return tokens[:, :-1], tokens[:, 1:], choice_target_mask[:, 1:]
