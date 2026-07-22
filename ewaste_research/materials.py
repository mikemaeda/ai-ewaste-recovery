"""Material database loading and recovery score calculations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .taxonomy import normalize_component

PRECIOUS_METALS = {
    "gold",
    "silver",
    "palladium",
    "platinum",
    "rare earth",
    "neodymium",
    "dysprosium",
}


@dataclass(frozen=True)
class RecoveryFeatures:
    value_usd_per_kg: float
    copper_percent: float
    precious_metal_percent: float
    recovery_score: float


def load_material_database(path: Path | str = "material_values.json") -> dict:
    """Load the JSON material database."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def get_component_entry(database: dict, label: str) -> dict | None:
    """Return a rich material entry for a component label, if one exists."""
    canonical = normalize_component(label)
    entry = database.get(canonical)
    return entry if isinstance(entry, dict) else None


def material_percent(entry: dict, material_keywords: set[str]) -> float:
    """Sum percentages for materials whose names include any keyword."""
    total = 0.0
    for material in entry.get("materials", []):
        name = str(material.get("material", "")).lower()
        if any(keyword in name for keyword in material_keywords):
            total += float(material.get("percent", 0.0))
    return total


def calculate_recovery_features(entry: dict | None) -> RecoveryFeatures:
    """
    Calculate a normalized 0-100 recovery score.

    Weights reflect the prototype's research focus: economic value matters most,
    copper is a major recoverable base metal, and precious metals add strategic
    value even at low mass percentages.
    """
    if not entry:
        return RecoveryFeatures(0.0, 0.0, 0.0, 0.0)

    value = float(entry.get("estimated_value_usd_per_kg", 0.0))
    copper = material_percent(entry, {"copper"})
    precious = material_percent(entry, PRECIOUS_METALS)

    value_score = min(value / 40.0, 1.0) * 100.0
    copper_score = min(copper / 65.0, 1.0) * 100.0
    precious_score = min(precious / 0.25, 1.0) * 100.0
    score = (0.50 * value_score) + (0.30 * copper_score) + (0.20 * precious_score)
    return RecoveryFeatures(value, copper, precious, round(score, 1))


def recommended_priority(score: float, configured_priority: str | None = None) -> str:
    """Convert a recovery score into a human-readable recovery priority."""
    if score >= 70:
        calculated = "critical"
    elif score >= 45:
        calculated = "high"
    elif score >= 25:
        calculated = "medium"
    else:
        calculated = "low"

    priority_rank = {"low": 0, "medium": 1, "high": 2, "critical": 3}
    configured = (configured_priority or "").lower()
    if configured in priority_rank and priority_rank[configured] > priority_rank[calculated]:
        return configured
    return calculated
