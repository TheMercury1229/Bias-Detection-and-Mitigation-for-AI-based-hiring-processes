"""Schema validation utilities for fairness-aware hiring datasets."""
from __future__ import annotations
from typing import Iterable, Sequence
import pandas as pd


def validate_dataset_schema(
        df: pd.DataFrame,
        required_columns: Sequence[str],
        target_column: str,
        sensitive_attributes: Iterable[str],
) -> dict[str, object]:
    """Validate schema requirements for a hiring fairness dataset.

    Args:
            df: Dataset to validate.
            required_columns: Feature columns that must be present (for example,
                    experience and education).
            target_column: Column representing hiring outcome.
            sensitive_attributes: One or more sensitive attribute candidates
                    (for example, gender, race, age).

    Returns:
            A summary dictionary containing:
                    - shape: tuple[int, int] with (rows, columns)
                    - present_sensitive_attributes: list[str] found in the dataset

    Raises:
            ValueError: If dataset is empty or validation constraints are not met.
    """
    if df is None:
        raise ValueError("Dataset is None. Expected a pandas DataFrame.")

    if df.empty:
        raise ValueError("Dataset is empty. Provide a non-empty dataset.")

    missing_required = [
        col for col in required_columns if col not in df.columns]
    if missing_required:
        raise ValueError(
            "Missing required feature columns: " + ", ".join(missing_required)
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' is missing from the dataset."
        )

    sensitive_attributes = list(sensitive_attributes)
    if not sensitive_attributes:
        raise ValueError(
            "At least one sensitive attribute name must be provided.")

    present_sensitive_attributes = [
        attr for attr in sensitive_attributes if attr in df.columns
    ]
    if not present_sensitive_attributes:
        raise ValueError(
            "No sensitive attribute column found. Expected at least one of: "
            + ", ".join(sensitive_attributes)
        )

    return {
        "shape": df.shape,
        "present_sensitive_attributes": present_sensitive_attributes,
    }
