"""OpenCV dashboard rendering for live e-waste recovery inference."""

from __future__ import annotations

import textwrap

import cv2
import numpy as np

from .materials import calculate_recovery_features, recommended_priority


def _put_text(panel, text: str, x: int, y: int, scale=0.48, color=(235, 235, 235)):
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def render_dashboard(frame, detections: list[dict], values: dict):
    """Return frame with a right-side dashboard containing material/recovery data."""
    height = frame.shape[0]
    panel_width = 430
    panel = np.full((height, panel_width, 3), (26, 31, 36), dtype=np.uint8)

    _put_text(panel, "E-Waste Recovery Dashboard", 18, 30, 0.62, (255, 255, 255))
    _put_text(panel, f"Detections: {len(detections)}", 18, 56, 0.48, (180, 215, 255))

    if not detections:
        _put_text(panel, "No component above threshold", 18, 100, 0.48, (180, 180, 180))
        return np.hstack([frame, panel])

    ranked_detections = sorted(
        detections,
        key=lambda detection: calculate_recovery_features(
            values.get(detection["label"]) if isinstance(values.get(detection["label"]), dict) else None
        ).recovery_score,
        reverse=True,
    )

    y = 92
    for rank, detection in enumerate(ranked_detections[:4], start=1):
        label = detection["label"]
        confidence = detection["confidence"]
        entry = values.get(label)
        features = calculate_recovery_features(entry if isinstance(entry, dict) else None)
        priority = recommended_priority(
            features.recovery_score,
            entry.get("recovery_priority") if isinstance(entry, dict) else None,
        )
        display = entry.get("display_name", label) if isinstance(entry, dict) else label

        cv2.rectangle(panel, (14, y - 22), (panel_width - 14, min(y + 122, height - 12)), (45, 54, 62), -1)
        _put_text(panel, f"#{rank} {display}  {confidence:.0%}", 24, y, 0.52, (255, 255, 255))
        _put_text(panel, f"Value: ${features.value_usd_per_kg:.2f}/kg", 24, y + 24)
        _put_text(panel, f"Copper: {features.copper_percent:.2f}%   Precious: {features.precious_metal_percent:.3f}%", 24, y + 46)
        _put_text(panel, f"Recovery score: {features.recovery_score:.1f}/100", 24, y + 68, 0.5, (122, 220, 145))
        _put_text(panel, f"Priority: {priority}", 24, y + 90, 0.5, (122, 220, 145))

        if isinstance(entry, dict):
            materials = sorted(entry.get("materials", []), key=lambda m: m.get("percent", 0), reverse=True)[:3]
            summary = ", ".join(f"{m.get('material')}: {m.get('percent')}%" for m in materials)
            for line in textwrap.wrap(summary, width=48)[:2]:
                _put_text(panel, line, 24, y + 112, 0.42, (205, 205, 205))
                y += 18

        y += 140
        if y > height - 120:
            break

    return np.hstack([frame, panel])
