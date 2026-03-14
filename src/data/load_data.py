"""Utilities for loading tabular hiring datasets."""

from __future__ import annotations
import pandas as pd


def load_dataset(file_path: str, verbose: bool = True) -> pd.DataFrame:
    """Load a hiring dataset from a CSV file.

    Args:
            file_path: Path to the CSV file.
            verbose: If True, prints dataset shape information.

    Returns:
            A pandas DataFrame containing the dataset.

    Raises:
            FileNotFoundError: If the provided file path does not exist.
            pd.errors.EmptyDataError: If the CSV file has no data.
            ValueError: If the loaded dataset has no rows.
    """
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"Dataset loaded from '{file_path}' is empty.")

    if verbose:
        rows, cols = df.shape
        print(f"Dataset loaded successfully: {rows} rows x {cols} columns")

    return df
