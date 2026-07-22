"""Rank e-waste classes by recovery score and export a poster-ready table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ewaste_research.materials import (
    calculate_recovery_features,
    load_material_database,
    recommended_priority,
)
from ewaste_research.taxonomy import CANONICAL_CLASSES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank component classes by recovery value.")
    parser.add_argument("--values", type=Path, default=Path("material_values.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reports/recovery_priority_table.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    database = load_material_database(args.values)
    rows = []

    for label in CANONICAL_CLASSES:
        entry = database.get(label)
        if not isinstance(entry, dict):
            continue
        features = calculate_recovery_features(entry)
        rows.append(
            {
                "class": label,
                "display_name": entry.get("display_name", label),
                "estimated_value_usd_per_kg": features.value_usd_per_kg,
                "copper_percent": features.copper_percent,
                "precious_metal_percent": features.precious_metal_percent,
                "recovery_score": features.recovery_score,
                "recommended_priority": recommended_priority(
                    features.recovery_score,
                    entry.get("recovery_priority"),
                ),
            }
        )

    rows.sort(key=lambda row: row["recovery_score"], reverse=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print("Recovery priority ranking:")
    for index, row in enumerate(rows[:10], start=1):
        print(
            f"{index:>2}. {row['display_name']}: score={row['recovery_score']:.1f}, "
            f"value=${row['estimated_value_usd_per_kg']:.2f}/kg, "
            f"priority={row['recommended_priority']}"
        )
    print(f"\nTable saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()

