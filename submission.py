"""Submission validation, format reserve generation, atomic publishing, and candidate storage."""

from dataclasses import dataclass
import os
from pathlib import Path
import shutil
import tempfile
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from experiments import TaskSpec


def validate_submission(
    csv_path: Path, task_spec: Any, test_path: Path
) -> tuple[bool, str]:
    """Strictly validate prediction CSV against test data and task specification."""
    if not csv_path.exists():
        return False, f"File does not exist: {csv_path}"

    if csv_path.stat().st_size == 0:
        return False, f"File is empty: {csv_path}"

    try:
        sub_df = pd.read_csv(csv_path)
    except Exception as exc:
        return False, f"Cannot read CSV: {exc}"

    expected_cols = [task_spec.id_column, task_spec.target_column]
    if list(sub_df.columns) != expected_cols:
        return (
            False,
            f"Columns mismatch: expected {expected_cols}, got {list(sub_df.columns)}",
        )

    # Validate against test.csv IDs
    try:
        test_df = pd.read_csv(test_path, usecols=[task_spec.id_column])
    except Exception as exc:
        return False, f"Cannot read test.csv for ID validation: {exc}"

    if len(sub_df) != len(test_df):
        return (
            False,
            f"Row count mismatch: expected {len(test_df)}, got {len(sub_df)}",
        )

    # Check ID match and order
    if not (sub_df[task_spec.id_column].values == test_df[task_spec.id_column].values).all():
        return False, "ID mismatch or order differs from test.csv"

    # Check for NaN / null
    target_series = sub_df[task_spec.target_column]
    if target_series.isna().any():
        return False, f"NaN / null values found in {task_spec.target_column}"

    # Check values by task type
    if task_spec.problem_type == "regression":
        if not pd.api.types.is_numeric_dtype(target_series.dtype):
            try:
                numeric_vals = pd.to_numeric(target_series)
            except Exception:
                return False, "Target values are not numeric for regression"
            if not np.isfinite(numeric_vals).all():
                return False, "Non-finite (inf/NaN) values found in regression target"
        elif not np.isfinite(target_series).all():
            return False, "Non-finite (inf/NaN) values found in regression target"
    else:
        # Classification
        if task_spec.classes:
            allowed = set(task_spec.classes)
            actual = set(target_series.astype(str).unique())
            invalid = actual - allowed
            if invalid:
                return (
                    False,
                    f"Invalid class label(s) for classification: {list(invalid)[:5]}",
                )

    return True, "Valid submission"


def create_reserve_submission(
    task_spec: Any,
    test_path: Path,
    sample_path: Path | None,
    out_path: Path,
) -> Path:
    """Generate safe format reserve without model training or label peeking."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if sample_path and sample_path.exists():
        try:
            sample_df = pd.read_csv(sample_path)
            expected_cols = [task_spec.id_column, task_spec.target_column]
            if list(sample_df.columns) == expected_cols:
                test_df = pd.read_csv(test_path, usecols=[task_spec.id_column])
                if len(sample_df) == len(test_df) and (
                    sample_df[task_spec.id_column].values == test_df[task_spec.id_column].values
                ).all():
                    # Direct copy of sample
                    sample_df.to_csv(out_path, index=False)
                    return out_path
        except Exception:
            pass

    # Fallback to constant
    test_df = pd.read_csv(test_path, usecols=[task_spec.id_column])
    constant_val = 0.0 if task_spec.problem_type == "regression" else (
        task_spec.classes[0] if task_spec.classes else "0"
    )
    reserve_df = pd.DataFrame(
        {
            task_spec.id_column: test_df[task_spec.id_column],
            task_spec.target_column: constant_val,
        }
    )
    reserve_df.to_csv(out_path, index=False)
    return out_path


def atomic_publish(source_csv: Path, target_path: Path) -> None:
    """Atomically copy/replace source_csv into target_path on the same filesystem."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_file = target_path.with_name(f".tmp_{os.getpid()}_{target_path.name}")

    try:
        shutil.copyfile(source_csv, temp_file)
        # Flush to disk
        with open(temp_file, "r+b") as f:
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_file, target_path)
    finally:
        if temp_file.exists():
            try:
                temp_file.unlink()
            except Exception:
                pass


@dataclass
class CandidateRecord:
    candidate_id: int
    csv_path: Path
    metric_value: float
    is_valid: bool


class CandidateStore:
    """Stores validated predictions for candidates and manages atomic publishing."""

    def __init__(self, candidates_dir: Path, target_submission: Path):
        self.candidates_dir = candidates_dir
        self.target_submission = target_submission
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.candidates: list[CandidateRecord] = []
        self.best_candidate: CandidateRecord | None = None

    def save_candidate(
        self, candidate_id: int, source_csv: Path, metric_value: float, task_spec: Any
    ) -> CandidateRecord:
        """Store candidate CSV and record entry."""
        cand_path = self.candidates_dir / f"candidate_{candidate_id}.csv"
        shutil.copyfile(source_csv, cand_path)

        record = CandidateRecord(
            candidate_id=candidate_id,
            csv_path=cand_path,
            metric_value=metric_value,
            is_valid=True,
        )
        self.candidates.append(record)

        # Update best candidate
        if self.best_candidate is None:
            self.best_candidate = record
        else:
            is_better = (
                metric_value < self.best_candidate.metric_value
                if task_spec.metric_direction == "minimize"
                else metric_value > self.best_candidate.metric_value
            )
            if is_better:
                self.best_candidate = record

        return record

    def publish_best(self) -> bool:
        """Atomically publish the best candidate CSV to final target."""
        if self.best_candidate and self.best_candidate.csv_path.exists():
            atomic_publish(self.best_candidate.csv_path, self.target_submission)
            return True
        return False
