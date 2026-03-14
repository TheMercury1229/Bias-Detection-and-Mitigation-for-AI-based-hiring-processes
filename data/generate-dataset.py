"""Generate a synthetic hiring dataset with intentional gender bias.

Usage:
	python data/generate-dataset.py
	python data/generate-dataset.py --rows 1000 --output data/hiring_synthetic_biased.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EDUCATION_LEVELS = ["Bachelors", "Masters", "PhD"]


def _sample_education(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample education levels with realistic proportions."""
    return rng.choice(EDUCATION_LEVELS, size=size, p=[0.60, 0.30, 0.10])


def _simulate_hiring_probability(
        gender: np.ndarray,
        age: np.ndarray,
        education_level: np.ndarray,
        years_experience: np.ndarray,
        skills_score: np.ndarray,
        interview_score: np.ndarray,
        university_rank: np.ndarray,
        employment_gap: np.ndarray,
) -> np.ndarray:
    """Convert candidate profile into hiring probability with intentional bias."""
    education_bonus_map = {
        "Bachelors": 0.00,
        "Masters": 0.08,
        "PhD": 0.14,
    }
    education_bonus = np.array([education_bonus_map[level]
                               for level in education_level])

    # Merit-driven score (both genders get similar feature distributions).
    merit_score = (
        0.33 * (skills_score / 100.0)
        + 0.37 * (interview_score / 100.0)
        + 0.14 * np.clip(years_experience / 12.0, 0.0, 1.0)
        + 0.10 * (4 - university_rank) / 3.0
        + 0.06 * (1 - employment_gap)
        + education_bonus
    )

    # Mild age effect around typical hiring preference band.
    age_effect = -0.0012 * np.abs(age - 32)

    # Intentional discriminatory boost for male candidates.
    gender_bias = np.where(gender == "Male", 0.10, -0.02)

    logit = -0.95 + merit_score + age_effect + gender_bias
    probability = 1.0 / (1.0 + np.exp(-6.5 * (logit - 0.5)))

    # Clip to keep probabilities in a realistic band.
    return np.clip(probability, 0.03, 0.97)


def generate_synthetic_hiring_dataset(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hiring data with intentional gender bias in outcomes."""
    if rows <= 0:
        raise ValueError("rows must be greater than 0.")

    rng = np.random.default_rng(seed)

    candidate_id = np.arange(1, rows + 1)
    gender = rng.choice(["Male", "Female"], size=rows, p=[0.5, 0.5])

    # Similar distributions across groups by design.
    age = np.clip(rng.normal(loc=31, scale=6.5,
                  size=rows).round(), 21, 55).astype(int)
    education_level = _sample_education(rng, rows)
    years_experience = np.clip(
        age - 21 + rng.normal(0, 2.3, size=rows), 0, 30).round(1)
    skills_score = np.clip(rng.normal(
        loc=69, scale=13, size=rows), 20, 100).round(1)
    interview_score = np.clip(
        0.55 * skills_score + rng.normal(loc=29, scale=12, size=rows),
        0,
        100,
    ).round(1)
    university_rank = rng.choice([1, 2, 3], size=rows, p=[0.24, 0.46, 0.30])
    employment_gap = rng.binomial(n=1, p=0.19, size=rows)

    hiring_probability = _simulate_hiring_probability(
        gender=gender,
        age=age,
        education_level=education_level,
        years_experience=years_experience,
        skills_score=skills_score,
        interview_score=interview_score,
        university_rank=university_rank,
        employment_gap=employment_gap,
    )
    hired = rng.binomial(n=1, p=hiring_probability, size=rows)

    df = pd.DataFrame(
        {
            "candidate_id": candidate_id,
            "gender": gender,
            "age": age,
            "education_level": education_level,
            "years_experience": years_experience,
            "skills_score": skills_score,
            "interview_score": interview_score,
            "university_rank": university_rank,
            "employment_gap": employment_gap,
            "hired": hired,
        }
    )

    return df


def save_dataset(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save generated dataset to CSV."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic hiring dataset with intentional gender bias."
    )
    parser.add_argument("--rows", type=int, default=1000,
                        help="Number of rows to generate.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument(
        "--output",
        default="data/hiring_synthetic_biased.csv",
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = generate_synthetic_hiring_dataset(rows=args.rows, seed=args.seed)
    output_path = save_dataset(df, args.output)

    gender_summary = (
        df.groupby("gender", dropna=False)[
            "hired"].mean().rename("hiring_rate")
    )

    print(f"Generated dataset with {len(df)} rows.")
    print(f"Saved to: {output_path.resolve()}")
    print("Hiring rate by gender (intentional bias expected):")
    print(gender_summary.to_string())


if __name__ == "__main__":
    main()
