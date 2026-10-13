"""Cached PyTorch datasets for preprocessed MRIxFields NPZ slices.

Every file read is routed through the official MRIxFields audit wrapper before
NumPy opens the file. The optional subject filters are used to create explicit,
subject-disjoint train/validation partitions for the audited Task 3 workflow.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from audit_dataset import audited_npz_image

from .transforms import CenterCropOrPad, ToTensor, ScaleToMinusOneOne, Compose
from .utils import FIELD_STRENGTHS, FIELD_TO_DOMAIN, MODALITIES, get_joint_domain


def _cached_transform(crop_size: Optional[Tuple[int, int]] = None) -> Compose:
    """Transform cached data from ``[0, 1]`` into tensors in ``[-1, 1]``."""
    steps = []
    if crop_size is not None:
        steps.append(CenterCropOrPad(crop_size))
    steps += [ToTensor(), ScaleToMinusOneOne()]
    return Compose(steps)


def _list_npz_files(
    base_dir: Path,
    split: str,
    modality: str,
    field: str,
) -> List[Path]:
    directory = base_dir / split / modality / field
    if not directory.exists():
        return []
    return sorted(directory.glob("*.npz"))


def cached_subject_id(
    path: str | Path,
    split: str,
    modality: str,
    field: str,
) -> str:
    """Extract the complete subject ID from a generated cached filename.

    The preprocessing filename is
    ``{split}_{modality}_{field}_{subject_id}_s{index}.npz``. Subject IDs may
    themselves contain underscores, so fixed-position splitting is avoided.
    """
    path = Path(path)
    prefix = f"{split}_{modality}_{field}_"
    stem = path.stem
    if stem.startswith(prefix):
        remainder = stem[len(prefix):]
    else:
        remainder = stem
    if "_s" in remainder:
        return remainder.rsplit("_s", 1)[0]
    return remainder


def _subject_allowed(
    subject_id: str,
    include_subjects: Optional[Iterable[str]],
    exclude_subjects: Optional[Iterable[str]],
) -> bool:
    include = set(include_subjects) if include_subjects is not None else None
    exclude = set(exclude_subjects or [])
    if include is not None and subject_id not in include:
        return False
    return subject_id not in exclude


def _filter_files_by_subject(
    files: Iterable[Path],
    *,
    split: str,
    modality: str,
    field: str,
    include_subjects: Optional[Iterable[str]] = None,
    exclude_subjects: Optional[Iterable[str]] = None,
) -> List[Path]:
    return [
        path
        for path in files
        if _subject_allowed(
            cached_subject_id(path, split, modality, field),
            include_subjects,
            exclude_subjects,
        )
    ]


class CachedUnpairedDataset(Dataset):
    """Single-domain cached slices used by CUT and CycleGAN."""

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: str,
        field_strength: str,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        include_subjects: Optional[Iterable[str]] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        self.modality = modality
        self.field_strength = field_strength
        self.transform = transform or _cached_transform(crop_size)

        files = _list_npz_files(
            self.preprocessed_dir, split, modality, field_strength
        )
        self.files = _filter_files_by_subject(
            files,
            split=split,
            modality=modality,
            field=field_strength,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
        )
        if not self.files:
            raise FileNotFoundError(
                f"No NPZ files found in "
                f"{self.preprocessed_dir / split / modality / field_strength}."
            )
        self.subject_ids = sorted(
            {
                cached_subject_id(path, split, modality, field_strength)
                for path in self.files
            }
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        image = audited_npz_image(self.files[index])
        if self.transform:
            image = self.transform(image)
        return {"image": image, "field_strength": self.field_strength}


class CachedPairedDataset(Dataset):
    """Subject- and slice-matched source/target cached slices."""

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: str,
        source_field: str,
        target_field: str,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
        include_subjects: Optional[Iterable[str]] = None,
        exclude_subjects: Optional[Iterable[str]] = None,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        self.split = split
        self.modality = modality
        self.source_field = source_field
        self.target_field = target_field
        self.transform = transform or _cached_transform(crop_size)

        source_files = _filter_files_by_subject(
            _list_npz_files(
                self.preprocessed_dir, split, modality, source_field
            ),
            split=split,
            modality=modality,
            field=source_field,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
        )
        target_files = _filter_files_by_subject(
            _list_npz_files(
                self.preprocessed_dir, split, modality, target_field
            ),
            split=split,
            modality=modality,
            field=target_field,
            include_subjects=include_subjects,
            exclude_subjects=exclude_subjects,
        )

        if not source_files or not target_files:
            raise FileNotFoundError(
                f"No cached files found for {source_field} or {target_field} "
                f"under {self.preprocessed_dir / split / modality}."
            )

        def pair_key(path: Path, field: str) -> Tuple[str, str]:
            subject = cached_subject_id(path, split, modality, field)
            slice_id = path.stem.rsplit("_", 1)[-1]
            return subject, slice_id

        target_lookup = {
            pair_key(path, target_field): path for path in target_files
        }
        self.pairs: List[Tuple[Path, Path]] = []
        self.pair_subjects: List[str] = []
        for source_path in source_files:
            key = pair_key(source_path, source_field)
            target_path = target_lookup.get(key)
            if target_path is not None:
                self.pairs.append((source_path, target_path))
                self.pair_subjects.append(key[0])

        if not self.pairs:
            raise ValueError(
                f"No matching pairs found between {source_field} and "
                f"{target_field}."
            )
        self.subject_ids = sorted(set(self.pair_subjects))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        source_path, target_path = self.pairs[index]
        source = audited_npz_image(source_path)
        target = audited_npz_image(target_path)
        if self.transform:
            source = self.transform(source)
            target = self.transform(target)
        return {"source": source, "target": target}


class CachedMultiDomainDataset(Dataset):
    """Multi-domain cached slices used by StarGAN v2."""

    def __init__(
        self,
        preprocessed_dir: str | Path,
        split: str,
        modality: Optional[str] = None,
        modalities: Optional[List[str]] = None,
        field_strengths: Optional[List[str]] = None,
        crop_size: Optional[Tuple[int, int]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        self.preprocessed_dir = Path(preprocessed_dir)
        if modalities is not None:
            self.modalities = list(modalities)
        elif modality is not None:
            self.modalities = [modality]
        else:
            self.modalities = list(MODALITIES)
        self.field_strengths = field_strengths or FIELD_STRENGTHS
        self.transform = transform or _cached_transform(crop_size)
        self._use_joint = len(self.modalities) > 1

        self.domain_files: Dict[int, List[Path]] = {}
        self.samples: List[Tuple[Path, int, str, str]] = []
        self._domain_to_pair: Dict[int, Tuple[str, str]] = {}

        for modality_name in self.modalities:
            for field in self.field_strengths:
                domain = (
                    get_joint_domain(modality_name, field)
                    if self._use_joint
                    else FIELD_TO_DOMAIN[field]
                )
                files = _list_npz_files(
                    self.preprocessed_dir, split, modality_name, field
                )
                if files:
                    self.domain_files.setdefault(domain, []).extend(files)
                    self._domain_to_pair[domain] = (modality_name, field)
                    self.samples.extend(
                        (path, domain, modality_name, field) for path in files
                    )

        if not self.samples:
            raise FileNotFoundError(
                f"No NPZ files found under {self.preprocessed_dir / split}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict:
        path, domain, modality, field = self.samples[index]
        image = audited_npz_image(path)

        other_domains = [value for value in self.domain_files if value != domain]
        reference_domain = random.choice(other_domains) if other_domains else domain
        reference_path = random.choice(self.domain_files[reference_domain])
        reference = audited_npz_image(reference_path)

        if self.transform:
            image = self.transform(image)
            reference = self.transform(reference)

        return {
            "image": image,
            "domain": domain,
            "field_strength": field,
            "modality": modality,
            "ref_image": reference,
            "ref_domain": reference_domain,
        }
