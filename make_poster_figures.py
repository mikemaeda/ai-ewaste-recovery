"""Generate all poster-ready figures for:

    AI-Assisted E-Waste Recovery Through Computer Vision and
    Material Composition Estimation

Every number is loaded from the project's own real artifacts:
  - material_values.json                                  (material database)
  - artifacts/reports/recovery_priority_table.csv         (recovery scores)
  - artifacts/reports/evaluation.json                     (classifier metrics)
  - artifacts/reports/dataset_analysis*/dataset_summary.json (dataset counts)

Output: artifacts/reports/poster/*.png  (300 DPI, poster-ready)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "artifacts" / "reports"
OUT = REPORTS / "poster"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Validated design tokens (from the data-viz reference palette)
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# Ordinal priority ramp (single-hue blue, light->dark = low->critical): CVD-safe
PRIORITY_COLOR = {
    "low": "#86b6ef",
    "medium": "#3987e5",
    "high": "#256abf",
    "critical": "#104281",
}
# Categorical hues (subset of the validated 8-hue reference set)
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_VIOLET = "#4a3aa7"
C_GOLD = "#eda100"
C_COPPER = "#d95926"
C_RED = "#e34948"


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.titlesize": 16,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": SECONDARY,
        "axes.edgecolor": BASELINE,
        "xtick.color": MUTED,
        "ytick.color": SECONDARY,
        "text.color": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.alpha": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def load_recovery_rows() -> list[dict]:
    rows = []
    with (REPORTS / "recovery_priority_table.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["estimated_value_usd_per_kg"] = float(r["estimated_value_usd_per_kg"])
            r["copper_percent"] = float(r["copper_percent"])
            r["precious_metal_percent"] = float(r["precious_metal_percent"])
            r["recovery_score"] = float(r["recovery_score"])
            rows.append(r)
    return rows


DB = json.loads((ROOT / "material_values.json").read_text(encoding="utf-8"))
ROWS = load_recovery_rows()
EVAL = json.loads((REPORTS / "evaluation.json").read_text(encoding="utf-8"))
DS_ORIG = json.loads((REPORTS / "dataset_analysis" / "dataset_summary.json").read_text(encoding="utf-8"))
DS_AUG = json.loads((REPORTS / "dataset_analysis_augmented_expanded" / "dataset_summary.json").read_text(encoding="utf-8"))


def save(fig, name: str) -> None:
    path = OUT / name
    fig.savefig(path, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  saved {path.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# FIG 1 -- Recovery Priority Score ranking  (the signature figure)
# ---------------------------------------------------------------------------
def fig_recovery_ranking() -> None:
    style()
    data = sorted(ROWS, key=lambda r: r["recovery_score"])
    names = [r["display_name"] for r in data]
    scores = [r["recovery_score"] for r in data]
    colors = [PRIORITY_COLOR[r["recommended_priority"]] for r in data]

    fig, ax = plt.subplots(figsize=(11, 12))
    bars = ax.barh(names, scores, color=colors, edgecolor=SURFACE, linewidth=1.2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Recovery priority score  (0 = 100)")
    ax.set_title("Component Recovery Priority Score", pad=14)
    ax.grid(axis="y", visible=False)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{s:.0f}", va="center", ha="left", fontsize=10,
                color=SECONDARY, fontweight="bold")

    handles = [mpatches.Patch(color=PRIORITY_COLOR[p], label=p.capitalize())
               for p in ["critical", "high", "medium", "low"]]
    ax.legend(handles=handles, title="Recommended priority", loc="lower right",
              frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.text(0, -1.4, "Score = 0.50 x economic value  +  0.30 x copper content  +  0.20 x precious-metal content",
            transform=ax.get_yaxis_transform(), fontsize=9.5, color=MUTED, ha="left")
    save(fig, "fig1_recovery_ranking.png")


# ---------------------------------------------------------------------------
# FIG 2 -- Recovery score as a weighted composition  (shows the algorithm)
# ---------------------------------------------------------------------------
def fig_score_composition() -> None:
    style()
    top = sorted(ROWS, key=lambda r: r["recovery_score"], reverse=True)[:12]
    top = top[::-1]
    names = [r["display_name"] for r in top]

    val_c, cu_c, pm_c = [], [], []
    for r in top:
        value_score = min(r["estimated_value_usd_per_kg"] / 40.0, 1.0) * 100.0
        copper_score = min(r["copper_percent"] / 65.0, 1.0) * 100.0
        precious_score = min(r["precious_metal_percent"] / 0.25, 1.0) * 100.0
        val_c.append(0.50 * value_score)
        cu_c.append(0.30 * copper_score)
        pm_c.append(0.20 * precious_score)

    fig, ax = plt.subplots(figsize=(11, 8))
    b1 = ax.barh(names, val_c, color=C_BLUE, edgecolor=SURFACE, linewidth=1.2,
                 label="Economic value  (weight 0.50)")
    b2 = ax.barh(names, cu_c, left=val_c, color=C_ORANGE, edgecolor=SURFACE,
                 linewidth=1.2, label="Copper content  (weight 0.30)")
    left2 = [a + b for a, b in zip(val_c, cu_c)]
    ax.barh(names, pm_c, left=left2, color=C_VIOLET, edgecolor=SURFACE,
            linewidth=1.2, label="Precious metals  (weight 0.20)")

    totals = [a + b + c for a, b, c in zip(val_c, cu_c, pm_c)]
    for y, t in enumerate(totals):
        ax.text(t + 1, y, f"{t:.0f}", va="center", ha="left",
                fontsize=10, color=SECONDARY, fontweight="bold")

    ax.set_xlim(0, 100)
    ax.set_xlabel("Contribution to recovery score")
    ax.set_title("How the Recovery Score Is Built (Top 12 Components)", pad=14)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID)
    save(fig, "fig2_score_composition.png")


# ---------------------------------------------------------------------------
# FIG 3 -- Economic value vs recovery score  (bubble = copper %)
# ---------------------------------------------------------------------------
def fig_value_vs_score() -> None:
    style()
    fig, ax = plt.subplots(figsize=(11, 8))
    for r in ROWS:
        color = PRIORITY_COLOR[r["recommended_priority"]]
        ax.scatter(r["estimated_value_usd_per_kg"], r["recovery_score"],
                   s=30 + r["copper_percent"] * 12, color=color, alpha=0.82,
                   edgecolor=SURFACE, linewidth=1.3, zorder=3)

    # annotate a few informative points
    annotate = {"cpu", "connector", "cable", "PCB", "transformer",
                "battery", "smartphone", "plastic_housing"}
    for r in ROWS:
        if r["class"] in annotate:
            ax.annotate(r["display_name"].split(" /")[0].split(" (")[0],
                        (r["estimated_value_usd_per_kg"], r["recovery_score"]),
                        textcoords="offset points", xytext=(8, 6),
                        fontsize=9, color=INK)

    ax.set_xlabel("Estimated economic value  (USD / kg)")
    ax.set_ylabel("Recovery priority score")
    ax.set_title("Economic Value vs. Recovery Score", pad=14)
    ax.text(0.02, 0.97,
            "Bubble size = recoverable copper %.\nCopper-rich but low-value parts (cable,\ntransformer) still earn a high score.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED, va="top")

    handles = [mpatches.Patch(color=PRIORITY_COLOR[p], label=p.capitalize())
               for p in ["critical", "high", "medium", "low"]]
    ax.legend(handles=handles, title="Priority", loc="lower right",
              frameon=True, facecolor=SURFACE, edgecolor=GRID)
    save(fig, "fig3_value_vs_score.png")


# ---------------------------------------------------------------------------
# FIG 4 -- Recoverable copper content by component
# ---------------------------------------------------------------------------
def fig_copper() -> None:
    style()
    data = sorted([r for r in ROWS if r["copper_percent"] > 0],
                  key=lambda r: r["copper_percent"])
    names = [r["display_name"] for r in data]
    cu = [r["copper_percent"] for r in data]

    fig, ax = plt.subplots(figsize=(10, 10))
    bars = ax.barh(names, cu, color=C_COPPER, edgecolor=SURFACE, linewidth=1.2)
    for bar, v in zip(bars, cu):
        ax.text(bar.get_width() + 0.6, bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}%", va="center", ha="left", fontsize=10,
                color=SECONDARY, fontweight="bold")
    ax.set_xlabel("Copper content  (% by mass)")
    ax.set_title("Recoverable Copper Content by Component", pad=14)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(cu) * 1.12)
    save(fig, "fig4_copper_content.png")


# ---------------------------------------------------------------------------
# FIG 5 -- Precious-metal content (gold / silver / palladium)
# ---------------------------------------------------------------------------
def fig_precious() -> None:
    style()
    order = ["cpu", "laptop_motherboard", "smartphone", "ram", "gpu",
             "connector", "PCB", "ssd", "tablet"]

    def pm(entry, kw):
        return sum(float(m["percent"]) for m in entry.get("materials", [])
                   if kw in m["material"].lower())

    labels, gold, silver, pall = [], [], [], []
    for key in order:
        e = DB[key]
        labels.append(e["display_name"].split(" /")[0].split(" (")[0])
        gold.append(pm(e, "gold"))
        silver.append(pm(e, "silver"))
        pall.append(pm(e, "palladium"))

    import numpy as np
    x = np.arange(len(labels))
    w = 0.26
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(x - w, gold, w, color=C_GOLD, edgecolor=SURFACE, linewidth=1, label="Gold")
    ax.bar(x, silver, w, color=C_BLUE, edgecolor=SURFACE, linewidth=1, label="Silver")
    ax.bar(x + w, pall, w, color=C_VIOLET, edgecolor=SURFACE, linewidth=1, label="Palladium")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("Content  (% by mass)")
    ax.set_title("Precious-Metal Content of High-Value Components", pad=14)
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID)
    save(fig, "fig5_precious_metals.png")


# ---------------------------------------------------------------------------
# FIG 6 -- Training dataset class distribution (after augmentation)
# ---------------------------------------------------------------------------
def fig_dataset() -> None:
    style()
    counts = DS_AUG["class_counts"]
    thr = DS_AUG["low_frequency_threshold"]
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in items]
    vals = [v for _, v in items]
    colors = [C_RED if v < thr else C_BLUE for v in vals]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(names, vals, color=colors, edgecolor=SURFACE, linewidth=1)
    ax.axhline(thr, color=INK, linestyle="--", linewidth=1.2,
               label=f"Low-frequency threshold ({thr})")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 60, f"{v:,}",
                ha="center", va="bottom", fontsize=9, color=SECONDARY)
    ax.set_ylabel("Images")
    total = DS_AUG["total_examples"]
    ax.set_title(f"Training Dataset After Class-Balancing Augmentation "
                 f"({total:,} images, {len(names)} classes)", pad=14, fontsize=14)
    ax.tick_params(axis="x", rotation=35)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=True, facecolor=SURFACE, edgecolor=GRID)
    ax.text(0.0, 1.005, f"Augmentation grew the corpus from {DS_ORIG['total_examples']:,} "
            f"to {total:,} images and removed all under-represented classes.",
            transform=ax.transAxes, fontsize=9.5, color=MUTED)
    save(fig, "fig6_dataset_distribution.png")


# ---------------------------------------------------------------------------
# FIG 7 -- Classifier results headline (hero number + stat tiles)
# ---------------------------------------------------------------------------
def fig_classifier_headline() -> None:
    style()
    acc = EVAL["validation_accuracy"] * 100
    fig, ax = plt.subplots(figsize=(13, 4.4))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # hero number (left third)
    ax.text(0.01, 0.60, f"{acc:.1f}%", fontsize=62, fontweight="bold",
            color=C_BLUE, ha="left", va="center")
    ax.text(0.01, 0.18, "MobileNetV2 validation accuracy", fontsize=14,
            color=SECONDARY, ha="left", va="center")

    tiles = [
        (f"{EVAL['epochs']}", "epochs"),
        (f"{len(EVAL['classes'])}", "classes"),
        (f"{DS_ORIG['total_examples']:,}", "images"),
        (f"{EVAL['training_images_percent']}/{EVAL['validation_images_percent']}", "train / val"),
        (f"{EVAL['validation_loss']:.3f}", "val. loss"),
    ]
    x0, pitch, bw = 0.375, 0.125, 0.112
    for i, (big, small) in enumerate(tiles):
        x = x0 + i * pitch
        cx = x + bw / 2
        ax.add_patch(FancyBboxPatch((x, 0.26), bw, 0.52,
                     boxstyle="round,pad=0.004,rounding_size=0.03",
                     transform=ax.transAxes, facecolor="#eef3fb",
                     edgecolor=GRID, linewidth=1))
        ax.text(cx, 0.60, big, fontsize=17, fontweight="bold",
                color=INK, ha="center", va="center")
        ax.text(cx, 0.37, small, fontsize=9.5, color=MUTED,
                ha="center", va="center")
    ax.set_title("Image Classification Results", loc="left", pad=10, fontsize=16)
    save(fig, "fig7_classifier_headline.png")


# ---------------------------------------------------------------------------
# FIG 8 -- System architecture / pipeline diagram
# ---------------------------------------------------------------------------
def fig_pipeline() -> None:
    style()
    fig, ax = plt.subplots(figsize=(6.5, 12))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 22)

    steps = [
        ("USB / Arducam Camera", "1280x720 @ 30 FPS live capture", C_BLUE),
        ("Computer Vision Model", "MobileNetV2 classifier  +  YOLO detector", C_BLUE),
        ("Component Classification", "predicted class  +  confidence", C_BLUE),
        ("Material Database Lookup", "copper, gold, silver, palladium, $/kg", C_ORANGE),
        ("Recovery Score", "0.50 value + 0.30 copper + 0.20 precious", C_ORANGE),
        ("Recovery Recommendation", "priority: critical / high / medium / low", C_VIOLET),
    ]
    n = len(steps)
    box_h, gap = 2.4, 1.05
    top = 21
    for i, (title, sub, color) in enumerate(steps):
        y = top - i * (box_h + gap)
        ax.add_patch(FancyBboxPatch((1.2, y - box_h), 7.6, box_h,
                     boxstyle="round,pad=0.05,rounding_size=0.3",
                     facecolor=SURFACE, edgecolor=color, linewidth=2.4))
        ax.add_patch(mpatches.Rectangle((1.2, y - box_h), 0.28, box_h,
                     facecolor=color, edgecolor="none"))
        ax.text(5.0, y - box_h * 0.36, title, fontsize=13.5, fontweight="bold",
                color=INK, ha="center", va="center")
        ax.text(5.0, y - box_h * 0.72, sub, fontsize=9.5, color=SECONDARY,
                ha="center", va="center")
        if i < n - 1:
            ay = y - box_h
            ax.add_patch(FancyArrowPatch((5.0, ay), (5.0, ay - gap),
                         arrowstyle="-|>", mutation_scale=20,
                         color=MUTED, linewidth=2))
    ax.set_title("End-to-End Recovery Pipeline", pad=12, fontsize=16)
    save(fig, "fig8_pipeline.png")


# ---------------------------------------------------------------------------
# FIG 9 -- Material composition table (top 10 recovery targets)
# ---------------------------------------------------------------------------
def fig_table() -> None:
    style()
    top = sorted(ROWS, key=lambda r: r["recovery_score"], reverse=True)[:10]
    col_labels = ["Component", "Value\n($/kg)", "Copper\n(%)",
                  "Precious\n(%)", "Score", "Priority"]
    cells, cell_colors = [], []
    for r in top:
        cells.append([
            r["display_name"],
            f"{r['estimated_value_usd_per_kg']:.1f}",
            f"{r['copper_percent']:.0f}",
            f"{r['precious_metal_percent']:.3f}",
            f"{r['recovery_score']:.0f}",
            r["recommended_priority"].capitalize(),
        ])
        pc = PRIORITY_COLOR[r["recommended_priority"]]
        cell_colors.append(["white"] * 5 + [pc])

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axis("off")
    tbl = ax.table(cellText=cells, colLabels=col_labels, cellColours=cell_colors,
                   cellLoc="center", loc="center",
                   colWidths=[0.30, 0.13, 0.13, 0.15, 0.10, 0.19])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.0)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        if row == 0:
            cell.set_facecolor("#eef3fb")
            cell.set_text_props(fontweight="bold", color=INK)
            cell.set_height(cell.get_height() * 1.25)
        else:
            if col == 0:
                cell.set_text_props(ha="left", color=INK)
                cell.PAD = 0.03
            if col == 5:  # priority pill -> white bold text
                cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("Top 10 Recovery Targets  -  Material Composition & Score",
                 pad=18, fontsize=15)
    save(fig, "fig9_material_table.png")


# ---------------------------------------------------------------------------
# FIG 10 -- Conclusion evidence snapshot
# ---------------------------------------------------------------------------
def fig_conclusion_snapshot() -> None:
    style()
    top = sorted(ROWS, key=lambda r: r["recovery_score"], reverse=True)[:6]
    top = top[::-1]
    names = [r["display_name"].split(" /")[0].split(" (")[0] for r in top]
    scores = [r["recovery_score"] for r in top]
    colors = [PRIORITY_COLOR[r["recommended_priority"]] for r in top]

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.15, 0.9],
                          height_ratios=[0.72, 0.28], wspace=0.34, hspace=0.28)
    ax_bar = fig.add_subplot(gs[:, :2])
    ax_note = fig.add_subplot(gs[0, 2])
    ax_stats = fig.add_subplot(gs[1, 2])

    bars = ax_bar.barh(names, scores, color=colors, edgecolor=SURFACE, linewidth=1.3)
    ax_bar.set_xlim(0, 85)
    ax_bar.set_xlabel("Recovery priority score")
    ax_bar.set_title("Highest-Priority Components for Recovery", loc="left", pad=14)
    ax_bar.grid(axis="y", visible=False)
    for bar, score in zip(bars, scores):
        ax_bar.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                    f"{score:.1f}", va="center", ha="left", fontsize=11,
                    color=SECONDARY, fontweight="bold")

    handles = [mpatches.Patch(color=PRIORITY_COLOR[p], label=p.capitalize())
               for p in ["critical", "high"]]
    ax_bar.legend(handles=handles, title="Priority", loc="lower right",
                  frameon=True, facecolor=SURFACE, edgecolor=GRID)

    ax_note.axis("off")
    ax_note.set_xlim(0, 1)
    ax_note.set_ylim(0, 1)
    ax_note.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.84,
                      boxstyle="round,pad=0.02,rounding_size=0.035",
                      facecolor="#eef3fb", edgecolor=GRID, linewidth=1.2))
    ax_note.text(0.08, 0.78, "Conclusion", fontsize=13, fontweight="bold",
                 color=INK, ha="left", va="center")
    ax_note.text(0.08, 0.54,
                 "Visual recognition becomes\nrecovery decision support\nwhen each detected part is\nlinked to material value.",
                 fontsize=12.5, color=SECONDARY, ha="left", va="center",
                 linespacing=1.28)
    ax_note.text(0.08, 0.22,
                 "Top targets: CPUs,\nconnectors, boards,\nphones, RAM, and GPUs.",
                 fontsize=10.5, color=MUTED, ha="left", va="center",
                 linespacing=1.18)

    ax_stats.axis("off")
    ax_stats.set_xlim(0, 1)
    ax_stats.set_ylim(0, 1)
    stats = [
        ("27", "components scored"),
        ("90.7%", "classifier accuracy"),
        ("3", "score inputs"),
    ]
    for i, (big, small) in enumerate(stats):
        x = 0.02 + i * 0.325
        ax_stats.add_patch(FancyBboxPatch((x, 0.10), 0.29, 0.78,
                           boxstyle="round,pad=0.012,rounding_size=0.03",
                           facecolor=SURFACE, edgecolor=GRID, linewidth=1.2))
        ax_stats.text(x + 0.145, 0.58, big, fontsize=19, fontweight="bold",
                      color=C_BLUE, ha="center", va="center")
        ax_stats.text(x + 0.145, 0.34, small, fontsize=8.7, color=MUTED,
                      ha="center", va="center")

    fig.suptitle("Conclusion: AI-Powered Sorting Can Prioritize Valuable E-Waste",
                 x=0.03, y=0.98, ha="left", fontsize=17, fontweight="bold")
    save(fig, "fig10_conclusion_snapshot.png")


if __name__ == "__main__":
    print("Generating poster figures from real project data ...")
    fig_recovery_ranking()
    fig_score_composition()
    fig_value_vs_score()
    fig_copper()
    fig_precious()
    fig_dataset()
    fig_classifier_headline()
    fig_pipeline()
    fig_table()
    fig_conclusion_snapshot()
    print(f"\nDone. {len(list(OUT.glob('*.png')))} figures in {OUT.relative_to(ROOT)}")
