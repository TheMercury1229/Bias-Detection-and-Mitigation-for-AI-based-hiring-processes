"""Generate a synthetic hiring dataset with intentional gender, age, and ethnic bias.

Usage:
	python data/generate-dataset.py
    python data/generate-dataset.py --rows 1000 --output data/hiring_synthetic_biased.csv

This dataset demonstrates:
- GENDER BIAS: +0.25 boost for males, -0.20 penalty for females
- AGE DISCRIMINATION: Strong preference for ages 28-38
- ETHNIC BIAS: Disparate impact across ethnic groups
- INTERSECTIONAL BIAS: Females with employment gaps face -0.12 penalty
- PROXY BIAS: Employment gap, university rank correlated with protected attributes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EDUCATION_LEVELS = ["Bachelors", "Masters", "PhD"]
ETHNICITIES = ["Caucasian", "Asian", "Hispanic", "African_American", "Other"]


def _sample_education(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample education levels with realistic proportions."""
    return rng.choice(EDUCATION_LEVELS, size=size, p=[0.60, 0.30, 0.10])


def _sample_ethnicity(rng: np.random.Generator, size: int) -> np.ndarray:
    """Sample ethnicity with realistic US demographic proportions."""
    return rng.choice(ETHNICITIES, size=size, p=[0.60, 0.18, 0.13, 0.07, 0.02])


def _simulate_hiring_probability(
        gender: np.ndarray,
        age: np.ndarray,
        ethnicity: np.ndarray,
        education_level: np.ndarray,
        years_experience: np.ndarray,
        skills_score: np.ndarray,
        interview_score: np.ndarray,
        university_rank: np.ndarray,
        employment_gap: np.ndarray,
) -> np.ndarray:
    """Convert candidate profile into hiring probability with INTENTIONAL BIAS.

    Bias mechanisms:
    1. GENDER: +0.25 male boost, -0.20 female penalty
    2. AGE: Strong preference ages 28-38 (ageism)
    3. ETHNICITY: Disparate impact favoring Caucasian/Asian
    4. INTERSECTIONAL: Females with gaps get additional -0.12 penalty
    5. PROXY: Employment gap and university rank correlations
    """
    education_bonus_map = {
        "Bachelors": 0.00,
        "Masters": 0.08,
        "PhD": 0.14,
    }
    education_bonus = np.array([education_bonus_map[level]
                               for level in education_level])

    # Merit-driven score (40% of base logit).
    merit_score = (
        0.28 * (skills_score / 100.0)
        + 0.32 * (interview_score / 100.0)
        + 0.12 * np.clip(years_experience / 12.0, 0.0, 1.0)
        + 0.10 * (4 - university_rank) / 3.0
        + 0.06 * (1 - employment_gap)
        + education_bonus
    )

    # **AGE DISCRIMINATION** - Strong penalty for <28 and >40.
    age_effect = -0.0020 * np.abs(age - 33)**1.5

    # **PRIMARY GENDER BIAS** - Explicit discrimination.
    gender_bias = np.where(gender == "Male", 0.25, -0.20)

    # **ETHNIC BIAS** - Disparate impact across ethnicities.
    ethnicity_bias = np.zeros(len(ethnicity))
    for i, eth in enumerate(ethnicity):
        if eth == "Caucasian":
            ethnicity_bias[i] = 0.15
        elif eth == "Asian":
            ethnicity_bias[i] = 0.12
        elif eth == "Hispanic":
            ethnicity_bias[i] = -0.08
        elif eth == "African_American":
            ethnicity_bias[i] = -0.12
        else:  # Other
            ethnicity_bias[i] = -0.10

    # **INTERSECTIONAL BIAS** - Females with employment gaps face compounding penalties.
    intersectional_penalty = np.where(
        (gender == "Female") & (employment_gap == 1), -0.12, 0.0
    )

    # **PROXY BIAS AMPLIFICATION** - University rank and employment gap matter more for some groups.
    proxy_amplification = np.where(
        gender == "Female",
        0.08 * (employment_gap + (3 - university_rank) / 3.0),  # More penalty
        # Slight benefit
        -0.03 * (employment_gap + (3 - university_rank) / 3.0)
    )

    # Combine all bias factors with a less extreme intercept/temperature so
    # positive outcomes are common enough for fairness comparisons.
    logit = (
        -0.65
        + merit_score
        + age_effect
        + gender_bias
        + ethnicity_bias
        + intersectional_penalty
        + proxy_amplification
    )

    # Convert logit to probability with sigmoid.
    probability = 1.0 / (1.0 + np.exp(-4.5 * (logit - 0.45)))

    # Clip to keep probabilities realistic but show clear disparate impact.
    return np.clip(probability, 0.05, 0.95)


def generate_synthetic_hiring_dataset(rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic hiring data with INTENTIONAL hierarchical bias.

    Bias dimensions:
    - Gender: 67% Male (representing historical overrepresentation)
    - Ethnicity: Caucasian-majority dataset
    - Age: Younger candidates preferred
    - Features: Correlated with protected attributes to show proxy bias
    """
    if rows <= 0:
        raise ValueError("rows must be greater than 0.")

    rng = np.random.default_rng(seed)

    candidate_id = np.arange(1, rows + 1)

    # **GENDER IMBALANCE** - Represent historical male overrepresentation (67% male).
    gender = rng.choice(["Male", "Female"], size=rows, p=[0.67, 0.33])

    # **ETHNICITY** - Skewed toward Caucasian (historical hiring patterns).
    ethnicity = _sample_ethnicity(rng, rows)

    # **DEMOGRAPHIC FEATURES** - Create dependencies on gender/ethnicity.
    age = np.clip(rng.normal(loc=31, scale=6.5,
                  size=rows).round(), 21, 60).astype(int)

    education_level = _sample_education(rng, rows)

    years_experience = np.clip(
        age - 21 + rng.normal(0, 2.3, size=rows), 0, 30).round(1)

    # **PROXY FEATURE 1: Skills Score** - Gender and ethnicity dependent.
    male_mask = gender == "Male"
    caucasian_mask = ethnicity == "Caucasian"

    skills_shift = np.where(male_mask, 3.5, -3.8)  # Stronger male advantage
    skills_shift += np.where(caucasian_mask, 2.0, -2.5)  # Ethnic penalty

    skills_score = np.clip(
        rng.normal(loc=69 + skills_shift, scale=13, size=rows), 20, 100
    ).round(1)

    # **INTERVIEW SCORE** - Correlated with skills but with additional bias.
    interview_score = np.clip(
        0.55 * skills_score + rng.normal(loc=29, scale=12, size=rows),
        0,
        100,
    ).round(1)

    # **PROXY FEATURE 2: University Rank** - Correlated with gender/ethnicity.
    university_rank = np.zeros(rows, dtype=int)
    for i in range(rows):
        if male_mask[i] and caucasian_mask[i]:
            university_rank[i] = rng.choice([1, 2, 3], p=[0.44, 0.38, 0.18])
        elif male_mask[i]:
            university_rank[i] = rng.choice([1, 2, 3], p=[0.25, 0.45, 0.30])
        elif caucasian_mask[i]:
            university_rank[i] = rng.choice([1, 2, 3], p=[0.18, 0.48, 0.34])
        else:
            university_rank[i] = rng.choice([1, 2, 3], p=[0.08, 0.40, 0.52])

    # **PROXY FEATURE 3: Employment Gap** - Gender and ethnicity correlated.
    employment_gap = np.zeros(rows, dtype=int)
    for i in range(rows):
        if male_mask[i]:
            employment_gap[i] = rng.binomial(n=1, p=0.10)
        elif caucasian_mask[i]:
            employment_gap[i] = rng.binomial(n=1, p=0.25)
        else:
            employment_gap[i] = rng.binomial(n=1, p=0.35)

    hiring_probability = _simulate_hiring_probability(
        gender=gender,
        age=age,
        ethnicity=ethnicity,
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
            "ethnicity": ethnicity,
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
        description="Generate a synthetic hiring dataset with INTENTIONAL hierarchical bias."
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

    print(f"\n{'='*70}")
    print(f"SYNTHETIC BIASED HIRING DATASET GENERATED")
    print(f"{'='*70}")
    print(f"Total records: {len(df)}")
    print(f"Output file: {output_path.resolve()}\n")

    # Gender bias analysis
    print("GENDER BIAS ANALYSIS:")
    gender_hire_rates = df.groupby("gender")["hired"].agg(["mean", "count"])
    gender_hire_rates.columns = ["hiring_rate", "count"]
    print(gender_hire_rates.to_string())
    male_rate = float(gender_hire_rates.loc["Male", "hiring_rate"])
    female_rate = float(gender_hire_rates.loc["Female", "hiring_rate"])
    disparate_impact_gender = (
        female_rate / male_rate if male_rate > 0 else float("inf")
    )
    if male_rate > 0:
        print(
            f"Disparate Impact Ratio (Female/Male): {disparate_impact_gender:.3f}")
    else:
        print("Disparate Impact Ratio (Female/Male): inf (male hiring rate is 0)")
    print(f"Expected bias: Female hiring rate ~35%, Male ~60%+\n")

    # Ethnicity bias analysis
    print("ETHNICITY BIAS ANALYSIS:")
    ethnicity_hire_rates = df.groupby(
        "ethnicity")["hired"].agg(["mean", "count"])
    ethnicity_hire_rates.columns = ["hiring_rate", "count"]
    print(ethnicity_hire_rates.to_string())
    print()

    # Age discrimination analysis
    print("AGE DISCRIMINATION ANALYSIS:")
    df["age_group"] = pd.cut(df["age"], bins=[20, 27, 38, 50, 61], labels=[
                             "21-27", "28-38", "39-50", "51+"])
    age_hire_rates = df.groupby("age_group")["hired"].agg(["mean", "count"])
    age_hire_rates.columns = ["hiring_rate", "count"]
    print(age_hire_rates.to_string())
    print(f"Expected bias: Ages 28-38 preferred (~50%+), others discriminated against\n")

    # Intersectional bias analysis
    print("INTERSECTIONAL BIAS ANALYSIS (Gender × Ethnicity):")
    intersectional = df.groupby(["gender", "ethnicity"])[
        "hired"].agg(["mean", "count"]).round(3)
    intersectional.columns = ["hiring_rate", "count"]
    print(intersectional.to_string())
    print(f"Expected: White males highest rate, minorities + females lowest\n")

    print("="*70)
    print("⚠️  DATASET INTENTIONALLY BIASED FOR FAIRNESS ANALYSIS")
    print("="*70)


if __name__ == "__main__":
    main()
