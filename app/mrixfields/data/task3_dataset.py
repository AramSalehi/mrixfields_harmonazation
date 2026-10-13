"""Audited Task 3 datasets for conditioned field-strength translation."""

from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset

from audit_dataset import audited_npz_image

from .cached_dataset import (
    _cached_transform,
    _list_npz_files,
    cached_subject_id,
)
from .dataset import _default_transform
from .utils import FIELD_STRENGTHS, MODALITIES, list_nifti_files, load_nifti


def _allowed(
    subject: str,
    include_subjects: Optional[Iterable[str]],
    exclude_subjects: Optional[Iterable[str]],
) -> bool:
    include = set(include_subjects) if include_subjects is not None else None
    exclude = set(exclude_subjects or [])
    return (include is None or subject in include) and subject not in exclude


class CachedTask3UnpairedDataset(Dataset):
    """Same-modality, different-field unpaired cached slices.

    ``deterministic=True`` is intended for validation. It maps every source
    slice to a stable target field and target slice using SHA-256 rather than
    the process-global random state.
    """

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modalities: Optional[List[str]] = None,
        field_strengths: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        include_subjects: Optional[Iterable[str]] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
        deterministic: bool = False,
        deterministic_seed: int = 2026,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        self.modalities = list(modalities or MODALITIES)
        self.field_strengths = list(field_strengths or FIELD_STRENGTHS)
        self.transform = transform or _cached_transform(crop_size)
        self.deterministic = bool(deterministic)
        self.deterministic_seed = int(deterministic_seed)

        self.domain_files: Dict[Tuple[str, str], List[Path]] = {}
        self.samples: List[Tuple[Path, str, str, str]] = []

        for modality in self.modalities:
            for field in self.field_strengths:
                files = []
                for path in _list_npz_files(
                    self.preprocessed_dir, split, modality, field
                ):
                    subject = cached_subject_id(path, split, modality, field)
                    if _allowed(subject, include_subjects, exclude_subjects):
                        files.append(path)
                        self.samples.append((path, modality, field, subject))
                if files:
                    self.domain_files[(modality, field)] = files

        if not self.samples:
            raise FileNotFoundError(
                f"No cached Task 3 slices found under "
                f"{self.preprocessed_dir / split}."
            )

        self.subject_ids = sorted({sample[3] for sample in self.samples})
        self.available_targets: Dict[Tuple[str, str], List[str]] = {}
        for _, modality, source_field, _ in self.samples:
            key = (modality, source_field)
            if key in self.available_targets:
                continue
            targets = [
                field
                for field in self.field_strengths
                if field != source_field
                and (modality, field) in self.domain_files
            ]
            if not targets:
                raise ValueError(
                    f"Task 3 needs at least two fields for modality={modality}."
                )
            self.available_targets[key] = targets

    def __len__(self) -> int:
        return len(self.samples)

    def _stable_index(self, source_path: Path, label: str, length: int) -> int:
        digest = hashlib.sha256(
            f"{self.deterministic_seed}|{source_path}|{label}".encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % length

    def __getitem__(self, index: int) -> Dict:
        source_path, modality, source_field, subject = self.samples[index]
        candidate_fields = self.available_targets[(modality, source_field)]

        if self.deterministic:
            field_index = self._stable_index(
                source_path, "target-field", len(candidate_fields)
            )
            target_field = candidate_fields[field_index]
            candidates = self.domain_files[(modality, target_field)]
            target_index = self._stable_index(
                source_path, f"target-path:{target_field}", len(candidates)
            )
            target_path = candidates[target_index]
        else:
            target_field = random.choice(candidate_fields)
            target_path = random.choice(
                self.domain_files[(modality, target_field)]
            )

        source = audited_npz_image(source_path)
        target = audited_npz_image(target_path)
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)

        return {
            "source": source,
            "target": target,
            "source_field": source_field,
            "target_field": target_field,
            "modality": modality,
            "subject_id": subject,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }


class Task3UnpairedMRIDataset(Dataset):
    """On-the-fly NIfTI fallback; every NIfTI read is officially audited."""

    def __init__(
        self,
        data_root: str | Path,
        split: str,
        modalities: Optional[List[str]] = None,
        field_strengths: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        slice_axis: int = 2,
        min_slice_std: float = 0.01,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.modalities = list(modalities or MODALITIES)
        self.field_strengths = list(field_strengths or FIELD_STRENGTHS)
        self.transform = transform or _default_transform(crop_size)
        self.slice_axis = int(slice_axis)
        self.min_slice_std = float(min_slice_std)

        self.domain_samples: Dict[
            Tuple[str, str], List[Tuple[Path, int]]
        ] = {}
        self.samples: List[Tuple[Path, int, str, str]] = []

        for modality in self.modalities:
            for field in self.field_strengths:
                key = (modality, field)
                indexed: List[Tuple[Path, int]] = []
                for path in list_nifti_files(
                    self.data_root, self.split, modality, field
                ):
                    volume, _ = load_nifti(path)
                    for slice_index in range(volume.shape[self.slice_axis]):
                        image = self._slice(volume, slice_index)
                        if image.std() > self.min_slice_std:
                            indexed.append((path, slice_index))
                            self.samples.append(
                                (path, slice_index, modality, field)
                            )
                if indexed:
                    self.domain_samples[key] = indexed

        if not self.samples:
            raise FileNotFoundError(
                f"No Task 3 NIfTI slices found under "
                f"{self.data_root / self.split}."
            )

        self.available_targets: Dict[Tuple[str, str], List[str]] = {}
        for _, _, modality, source_field in self.samples:
            key = (modality, source_field)
            if key in self.available_targets:
                continue
            targets = [
                field
                for field in self.field_strengths
                if field != source_field
                and (modality, field) in self.domain_samples
            ]
            if not targets:
                raise ValueError(
                    f"Task 3 needs at least two fields for modality={modality}."
                )
            self.available_targets[key] = targets

    def _slice(self, volume: np.ndarray, index: int) -> np.ndarray:
        slicing = [slice(None)] * volume.ndim
        slicing[self.slice_axis] = index
        return volume[tuple(slicing)]

    def _load_slice(self, path: Path, index: int):
        volume, _ = load_nifti(path)
        return self._slice(volume, index)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        source_path, source_index, modality, source_field = self.samples[index]
        target_field = random.choice(
            self.available_targets[(modality, source_field)]
        )
        target_path, target_index = random.choice(
            self.domain_samples[(modality, target_field)]
        )

        source = self._load_slice(source_path, source_index)
        target = self._load_slice(target_path, target_index)
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)

        return {
            "source": source,
            "target": target,
            "source_field": source_field,
            "target_field": target_field,
            "modality": modality,
            "source_path": str(source_path),
            "target_path": str(target_path),
        }


class ConditionedPairedDataset(Dataset):
    """Attach constant Task 3 condition metadata to a paired dataset."""

    def __init__(
        self,
        dataset: Dataset,
        modality: str,
        source_field: str,
        target_field: str,
    ) -> None:
        self.dataset = dataset
        self.modality = modality
        self.source_field = source_field
        self.target_field = target_field

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict:
        item = dict(self.dataset[index])
        item["modality"] = self.modality
        item["source_field"] = self.source_field
        item["target_field"] = self.target_field
        return item
