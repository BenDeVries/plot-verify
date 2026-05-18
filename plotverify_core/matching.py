"""Image/CSV pairing for the batch upload workflow.

Pairs files by canonical stem (case-insensitive, extension ignored). Reports
unmatched files and duplicate stems separately so the UI can prompt the user
to resolve ambiguity before processing.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FileEntry:
    """One uploaded file (image or CSV). The payload is held in memory."""
    filename: str            # original (case preserved)
    payload: bytes
    extension: str           # lowercase, with leading "." (e.g. ".png")
    canonical_stem: str      # lowercase stem (for matching)


def make_file_entry(filename: str, payload: bytes) -> FileEntry:
    """Build a FileEntry from a filename + bytes."""
    stem, ext = os.path.splitext(filename)
    return FileEntry(
        filename=filename,
        payload=payload,
        extension=ext.lower(),
        canonical_stem=stem.lower(),
    )


@dataclass
class MatchResult:
    """Outcome of `match_files`.

    Keys in `pairs` are canonical stems. Duplicate dicts map stem → list of
    conflicting entries (both lists have length ≥ 2).
    """
    pairs: Dict[str, Tuple[FileEntry, FileEntry]] = field(default_factory=dict)
    images_without_csv: List[FileEntry] = field(default_factory=list)
    csvs_without_image: List[FileEntry] = field(default_factory=list)
    duplicate_image_stems: Dict[str, List[FileEntry]] = field(default_factory=dict)
    duplicate_csv_stems: Dict[str, List[FileEntry]] = field(default_factory=dict)

    @property
    def has_duplicates(self) -> bool:
        return bool(self.duplicate_image_stems) or bool(self.duplicate_csv_stems)

    @property
    def has_unmatched(self) -> bool:
        return bool(self.images_without_csv) or bool(self.csvs_without_image)

    @property
    def ok_to_proceed(self) -> bool:
        """True when at least one matched pair exists and no duplicates remain."""
        return bool(self.pairs) and not self.has_duplicates


def match_files(images: List[FileEntry], csvs: List[FileEntry]) -> MatchResult:
    """Match image/CSV pairs by canonical (lowercase) stem.

    Files with the same stem appearing more than once within their group are
    reported in `duplicate_*_stems` and excluded from `pairs`.
    """
    result = MatchResult()

    # Group by stem within each input list.
    by_stem_img: Dict[str, List[FileEntry]] = {}
    for img in images:
        by_stem_img.setdefault(img.canonical_stem, []).append(img)
    by_stem_csv: Dict[str, List[FileEntry]] = {}
    for csv in csvs:
        by_stem_csv.setdefault(csv.canonical_stem, []).append(csv)

    # Detect duplicates within each group; remove from candidate pool.
    for stem, items in list(by_stem_img.items()):
        if len(items) > 1:
            result.duplicate_image_stems[stem] = items
            del by_stem_img[stem]
    for stem, items in list(by_stem_csv.items()):
        if len(items) > 1:
            result.duplicate_csv_stems[stem] = items
            del by_stem_csv[stem]

    # Pair remaining unique-stem entries. Sort all stem iterations for
    # stable, reproducible output ordering — important for both the UI
    # (so file lists don't reshuffle between renders) and the tests.
    img_stems = set(by_stem_img)
    csv_stems = set(by_stem_csv)
    for stem in sorted(img_stems & csv_stems):
        result.pairs[stem] = (by_stem_img[stem][0], by_stem_csv[stem][0])

    for stem in sorted(img_stems - csv_stems):
        result.images_without_csv.append(by_stem_img[stem][0])
    for stem in sorted(csv_stems - img_stems):
        result.csvs_without_image.append(by_stem_csv[stem][0])

    return result


def find_pair(result: MatchResult, stem: str) -> Optional[Tuple[FileEntry, FileEntry]]:
    """Helper for callers that resolve stems case-insensitively."""
    return result.pairs.get(stem.lower())
