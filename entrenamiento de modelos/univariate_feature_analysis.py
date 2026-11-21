"""Compute multiclass univariate discrimination and predictor correlations."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import f_oneway


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def analyze(csv_path: Path, output_dir: Path, target: str, correlation_threshold: float) -> None:
    data = pd.read_csv(csv_path)
    if target not in data.columns:
        raise ValueError(f"Target column {target!r} was not found in {csv_path}")

    predictors = [column for column in data.columns if column != target]
    non_numeric = [column for column in predictors if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"Non-numeric predictors found: {non_numeric}")
    if data[predictors + [target]].isna().any().any():
        raise ValueError("The analysis expects complete cases; missing values were found.")

    classes = sorted(data[target].unique())
    if len(classes) < 2:
        raise ValueError("At least two target classes are required.")

    rows = []
    for feature in predictors:
        groups = [data.loc[data[target] == label, feature].to_numpy(dtype=float) for label in classes]
        statistic, p_value = f_oneway(*groups)
        values = data[feature].to_numpy(dtype=float)
        grand_mean = values.mean()
        ss_between = sum(len(group) * (group.mean() - grand_mean) ** 2 for group in groups)
        ss_total = ((values - grand_mean) ** 2).sum()
        eta_squared = ss_between / ss_total if ss_total else 0.0
        rows.append(
            {
                "feature": feature,
                "anova_f": float(statistic),
                "eta_squared": float(eta_squared),
                "p_value": float(p_value),
            }
        )

    discrimination = pd.DataFrame(rows).sort_values("eta_squared", ascending=False)
    correlations = data[predictors].corr(method="pearson")
    effect = discrimination.set_index("feature")["eta_squared"]

    pairs = []
    for first, second in combinations(predictors, 2):
        coefficient = float(correlations.loc[first, second])
        if abs(coefficient) >= correlation_threshold:
            pairs.append(
                {
                    "feature_1": first,
                    "feature_2": second,
                    "pearson_r": coefficient,
                    "feature_1_eta_squared": float(effect[first]),
                    "feature_2_eta_squared": float(effect[second]),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    discrimination.to_csv(output_dir / "univariate_discrimination.csv", index=False, float_format="%.10g")
    correlations.to_csv(output_dir / "predictor_correlations_pearson.csv", float_format="%.10g")
    pd.DataFrame(pairs).to_csv(output_dir / "high_correlation_pairs.csv", index=False, float_format="%.10g")

    print(f"Rows: {len(data)} | classes: {len(classes)} | predictors: {len(predictors)}")
    print(f"Discrimination results: {output_dir / 'univariate_discrimination.csv'}")
    print(f"Full Pearson matrix: {output_dir / 'predictor_correlations_pearson.csv'}")
    print(f"Pairs with |r| >= {correlation_threshold:g}: {len(pairs)}")
    for pair in pairs:
        print(f"  {pair['feature_1']} / {pair['feature_2']}: r={pair['pearson_r']:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=PROJECT_ROOT / "EngineFaultDB_Final.csv")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "resultados" / "analisis_individual")
    parser.add_argument("--target", default="Fault")
    parser.add_argument("--correlation-threshold", type=float, default=0.95)
    args = parser.parse_args()
    analyze(args.csv, args.output, args.target, args.correlation_threshold)


if __name__ == "__main__":
    main()
