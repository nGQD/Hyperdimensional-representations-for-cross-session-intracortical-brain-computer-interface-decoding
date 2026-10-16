"""Generate source-grounded vector figures for the H1 phase-HDC study.

Figures 1 and 2 are mechanism/protocol diagrams. Figures 3 and 4 are
empirical and deliberately require a complete, schema-validated
``results/summary.json``. This module never invents or fills in measurements.

Run from the repository root after the result record is complete:

    python -m hdc_study.graphics --summary hdc_study/results/summary.json

Vector EPS/PDF/SVG files are written to ``hdc_study/figures`` and PNG previews
to ``hdc_study/figures/previews``. Final PDFs are also copied to the manuscript
folder as ``Fig1.pdf`` through ``Fig4.pdf``. The artwork is sized for a 174 mm full-width
Springer figure; captions are kept outside the figure, in ``CAPTIONS.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
PREVIEW_DIR = FIG_DIR / "previews"
SUMMARY_DEFAULT = ROOT / "results" / "summary.json"

# Cardinal red / black / neutral palette, with distinct marker shapes and
# line styles so the plots remain readable when printed in grayscale.
INK = "#171717"
CARDINAL = "#A31F34"
CARDINAL_DARK = "#781426"
SLATE = "#45515E"
BLUE = "#3C78A8"
GOLD = "#D5A326"
TEAL = "#328A83"
PAPER = "#FFFFFF"
PALE = "#F3F4F5"
MID = "#D5D8DC"
MUTED = "#616A73"
COLORS = {"hdc": CARDINAL, "random": SLATE, "mlp": BLUE, "plain": TEAL,
          "selected_control": INK}
MARKERS = {"hdc": "o", "random": "s", "mlp": "^", "plain": "D",
           "selected_control": "P"}
FAMILY_LABELS = {
    "hdc": "HDC",
    "random": "Random projection",
    "mlp": "MLP features",
    "plain": "Contextual GRU",
    "selected_control": "Inner-selected control",
}


def _display_family(family: str) -> str:
    return FAMILY_LABELS.get(family, family.replace("_", " "))


def _mean_interpolated_kernel(distance: float, basis_kernel: np.ndarray,
                              levels: int = 8) -> float:
    """Average an anchor kernel over all valid pairs at a value difference."""
    anchors = np.linspace(0.0, 1.0, levels)
    basis_kernel = np.asarray(basis_kernel, dtype=np.float64)
    if basis_kernel.shape != (levels, levels):
        raise ValueError("basis_kernel must be square with one row per anchor")

    def anchor_weights(value: float) -> np.ndarray:
        scaled = value * (levels - 1)
        return np.maximum(0.0, 1.0 - np.abs(scaled - np.arange(levels)))

    end = 1.0 - distance
    if end <= 1e-12:
        return float(anchor_weights(0.0) @ basis_kernel @ anchor_weights(1.0))
    knots = {0.0, end}
    for anchor in anchors:
        if 0.0 < anchor < end:
            knots.add(float(anchor))
        shifted = float(anchor - distance)
        if 0.0 < shifted < end:
            knots.add(shifted)
    knots = sorted(knots)
    # On each interval, the two interpolated weight vectors are linear, so
    # their bilinear kernel is quadratic and three-point Gauss-Legendre is exact.
    nodes = (-math.sqrt(3.0 / 5.0), 0.0, math.sqrt(3.0 / 5.0))
    weights = (5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0)
    integral = 0.0
    for low, high in zip(knots[:-1], knots[1:]):
        midpoint = (low + high) / 2.0
        halfwidth = (high - low) / 2.0
        for node, weight in zip(nodes, weights):
            left = midpoint + halfwidth * node
            integral += halfwidth * weight * float(
                anchor_weights(left) @ basis_kernel @ anchor_weights(left + distance))
    return integral / end

# Fonts are embedded as TrueType in PDFs and retained as text in SVGs.
matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": PAPER,
})


def _panel_label(ax, label: str, x: float = -0.07, y: float = 1.07) -> None:
    ax.text(x, y, label, transform=ax.transAxes, fontsize=10, fontweight="bold",
            color=INK, ha="left", va="top", clip_on=False)


def _save(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", metadata={
        "Title": stem.replace("_", " "), "Creator": "hdc_study.graphics",
    })
    fig.savefig(FIG_DIR / f"{stem}.svg", metadata={
        "Title": stem.replace("_", " "), "Creator": "hdc_study.graphics",
    })
    fig.savefig(FIG_DIR / f"{stem}.eps", metadata={
        "Title": stem.replace("_", " "), "Creator": "hdc_study.graphics",
    })
    fig.savefig(PREVIEW_DIR / f"{stem}.png", dpi=220)
    figure_number = re.match(r"fig([1-4])_", stem)
    if figure_number:
        manuscript_dir = ROOT.parent / "neuroinformatics_submission"
        manuscript_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(FIG_DIR / f"{stem}.pdf",
                     manuscript_dir / f"Fig{figure_number.group(1)}.pdf")
    plt.close(fig)


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *,
           color: str = INK, lw: float = 1.2, style: str = "-|>") -> None:
    ax.annotate("", xy=end, xytext=start,
                arrowprops={"arrowstyle": style, "color": color, "lw": lw,
                            "shrinkA": 2, "shrinkB": 2})


def _box(ax, x: float, y: float, w: float, h: float, title: str,
         detail: str, face: str = PAPER, edge: str = INK,
         title_color: str = INK) -> None:
    shape = FancyBboxPatch((x, y), w, h,
                           boxstyle="round,pad=0.012,rounding_size=0.02",
                           facecolor=face, edgecolor=edge, linewidth=1.15)
    ax.add_patch(shape)
    ax.text(x + w / 2, y + h * .68, title, ha="center", va="center",
            fontsize=8.0, fontweight="bold", color=title_color)
    ax.text(x + w / 2, y + h * .32, detail, ha="center", va="center",
            fontsize=7.8, color=INK, linespacing=1.16)


def figure1_mechanism() -> None:
    """Architecture and exact theoretical properties; no measured traces."""
    fig = plt.figure(figsize=(174 / 25.4, 130 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")

    ax.text(.035, .99, "(a)", fontsize=10, weight="bold", color=INK, va="top")
    box_y, box_h = .765, .19
    boxes = (
        (.045, .12, "Normalized\nrates", r"$x_{t,1:C}$", PAPER, INK),
        (.19, .31, "Channel / value", r"$u=(1+\tanh x)/2$; 8 anchors" + "\n" +
         r"$q_t=C^{-1/2}\sum_{c,\ell}a_{t,c,\ell}$" + "\n" +
         r"$(r_c\odot v_\ell)$", "#F9EDEF", CARDINAL),
        (.53, .25, "Causal lag roles", r"$z_t=K^{-1/2}\sum_{j=0}^{K-1}$" + "\n" +
         r"$\rho_j\odot q_{t-j}$", "#F3F5F7", SLATE),
        (.81, .16, "GRU + readout", "learned projection" + "\n" +
         r"$\hat y_t\in\mathbb{R}^{7}$", "#F4F4F4", INK),
    )
    for x, width, title, detail, face, edge in boxes:
        ax.add_patch(FancyBboxPatch((x, box_y), width, box_h,
                                    boxstyle="round,pad=.008,rounding_size=.012",
                                    facecolor=face, edgecolor=edge, linewidth=1.0))
        ax.text(x + width / 2, box_y + .145, title, ha="center", va="center",
                fontsize=8.3, weight="bold", color=CARDINAL_DARK if edge == CARDINAL else edge)
        ax.text(x + width / 2, box_y + .067, detail, ha="center", va="center",
                fontsize=7.8, color=INK, linespacing=1.18)
    for start, end, color in (((.165, .84), (.19, .84), CARDINAL),
                              ((.50, .84), (.53, .84), SLATE),
                              ((.78, .84), (.81, .84), INK)):
        _arrow(ax, start, end, color=color, lw=1.0)

    # A unit-circle glyph illustrates one coordinate of the complex binding.
    phase_card = FancyBboxPatch((.19, .525), .31, .205,
                                boxstyle="round,pad=.008,rounding_size=.012",
                                facecolor=PAPER, edgecolor=MID, linewidth=.8)
    ax.add_patch(phase_card)
    ax.text(.205, .705,
            r"$\arg(r_c\odot v_\ell)\equiv\theta_c+\phi_\ell\;(\mathrm{mod}\;2\pi)$",
            fontsize=7.8, color=INK, va="center")
    phase_items = ((.245, r"$r_c$", 25, CARDINAL),
                   (.345, r"$v_\ell$", 50, BLUE),
                   (.445, r"$r_c\odot v_\ell$", 75, SLATE))
    for center_x, label, angle, color in phase_items:
        center = (center_x, .615)
        radius_y = .034
        radius_x = radius_y * 130 / 174
        ax.add_patch(Ellipse(center, 2 * radius_x, 2 * radius_y,
                             facecolor="none", edgecolor=MID, linewidth=.65))
        ax.plot([center_x - radius_x * .68, center_x + radius_x * .68],
                [center[1], center[1]], color=MID, linewidth=.45, zorder=1)
        ax.plot([center_x, center_x], [center[1] - radius_y * .68,
                                      center[1] + radius_y * .68],
                color=MID, linewidth=.45, zorder=1)
        endpoint = (center_x + radius_x * .82 * math.cos(math.radians(angle)),
                    center[1] + radius_y * .82 * math.sin(math.radians(angle)))
        ax.annotate("", xy=endpoint, xytext=center,
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.05},
                    zorder=3)
        ax.text(center_x, .562, label, ha="center", va="center",
                fontsize=7.8, color=color)

    ax.plot([.035, .965], [.495, .495], color=MID, linewidth=.8)
    ax.text(.035, .465, "(b)", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(.08, .472, "Expected value-code similarity", fontsize=8.7,
            weight="bold", color=INK, va="top")

    # These are exact expected kernels, evaluated by piecewise-exact Gaussian
    # quadrature on the same interpolation basis, not empirical curves.
    levels = 8
    anchors = np.linspace(0.0, 1.0, levels)
    anchor_kernel = np.exp(-2.0 * np.pi**2 *
                           (anchors[:, None] - anchors[None, :])**2)
    random_feature_kernel = np.eye(levels)

    distances = np.linspace(0.0, 1.0, 121)
    hdc_kernel = np.asarray([
        _mean_interpolated_kernel(float(delta), anchor_kernel, levels)
        for delta in distances
    ])
    random_kernel = np.asarray([
        _mean_interpolated_kernel(float(delta), random_feature_kernel, levels)
        for delta in distances
    ])
    hdc_normalized = hdc_kernel / hdc_kernel[0]
    random_normalized = random_kernel / random_kernel[0]
    kernel_ax = fig.add_axes([.095, .105, .54, .275])
    kernel_ax.plot(distances, hdc_normalized, color=CARDINAL, linewidth=2.0,
                   label="Correlated HDC anchors")
    kernel_ax.plot(distances, random_normalized, color=SLATE, linestyle="--",
                   linewidth=1.7, label="Gaussian random features")
    kernel_ax.set_xlim(0, 1)
    kernel_ax.set_ylim(0, 1.04)
    kernel_ax.set_xticks([0, .25, .5, .75, 1])
    kernel_ax.set_yticks([0, .25, .5, .75, 1])
    kernel_ax.set_xlabel(r"Bounded-value difference $|\Delta u|$", labelpad=2)
    kernel_ax.set_ylabel("Normalized expected similarity", labelpad=2)
    kernel_ax.tick_params(labelsize=7.8, length=2.5, pad=2)
    kernel_ax.grid(color=MID, linewidth=.45)
    kernel_ax.spines[["top", "right"]].set_visible(False)
    kernel_ax.legend(loc="upper right", frameon=False, fontsize=7.1,
                     handlelength=1.7, labelspacing=.35, borderaxespad=.15)
    ax.text(.095, .391,
            "Exact L=8 interpolation; average over valid value pairs",
            fontsize=7.8, color=MUTED, va="center")

    bound_x, bound_w = .69, .275
    bound_box = FancyBboxPatch((bound_x, .105), bound_w, .31,
                               boxstyle="round,pad=.01,rounding_size=.012",
                               facecolor=PALE, edgecolor=MID, linewidth=.8)
    ax.add_patch(bound_box)
    ax.text(bound_x + .018, .385, "Localized channel change", fontsize=8.5,
            weight="bold", color=INK, va="top")
    icon_x, icon_y, icon_w, icon_h, gap = bound_x + .025, .32, .020, .035, .006
    for index in range(8):
        color = CARDINAL if index < 2 else "#E1E4E7"
        edge = CARDINAL_DARK if index < 2 else SLATE
        ax.add_patch(Rectangle((icon_x + index * (icon_w + gap), icon_y),
                               icon_w, icon_h, facecolor=color,
                               edgecolor=edge, linewidth=.45))
    ax.text(icon_x + .02, .292, r"$m$ of $C$ changed (schematic)",
            fontsize=8.0, color=INK, va="top")
    ax.text(bound_x + .018, .232,
            r"$\mathbb{E}_r[\|\Delta q\|_2^2/D]\leq 4m/C$",
            fontsize=8.3, color=CARDINAL_DARK, va="center")
    ax.text(bound_x + .018, .176, "Roles averaged; fixed codebook",
            fontsize=7.8, color=INK, va="center")
    ax.text(bound_x + .018, .145, "Representation bound",
            fontsize=7.8, color=MUTED, va="center")
    _save(fig, "fig1_mechanism")


def figure2_protocol() -> None:
    """Compact nested-fold grid and group-level dose timeline."""
    fig = plt.figure(figsize=(174 / 25.4, 112 / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.text(.035, .965, "(a)", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(.075, .968, "Outer / inner group grid", fontsize=8.7,
            weight="bold", color=INK, va="top")

    # Rows are the outer group; columns are the inner validation group.
    # Symmetric off-diagonal cells share a fit on the other four groups.
    grid_x, grid_y, cell_w, cell_h = .105, .615, .042, .047
    for index in range(6):
        ax.text(grid_x + (index + .5) * cell_w, .914, f"S{index}",
                ha="center", va="center", fontsize=7.8, color=INK)
        ax.text(.073, grid_y + (5 - index + .5) * cell_h, f"S{index}",
                ha="center", va="center", fontsize=7.8, color=INK)
    for outer in range(6):
        for validation in range(6):
            x = grid_x + validation * cell_w
            y = grid_y + (5 - outer) * cell_h
            if outer == validation:
                ax.add_patch(Rectangle((x, y), cell_w, cell_h,
                                       facecolor="#E9ECEF", edgecolor=PAPER,
                                       linewidth=.8))
                ax.plot([x + .006, x + cell_w - .006],
                        [y + .006, y + cell_h - .006], color=MUTED, lw=.55)
                ax.plot([x + .006, x + cell_w - .006],
                        [y + cell_h - .006, y + .006], color=MUTED, lw=.55)
            else:
                ax.add_patch(Rectangle((x, y), cell_w, cell_h,
                                       facecolor="#F9EDEF", edgecolor=PAPER,
                                       linewidth=.8))
    ax.text(grid_x + 3 * cell_w, .592, "validation group →", ha="center",
            va="top", fontsize=7.8, color=MUTED)
    explanation = FancyBboxPatch((.405, .615), .56, .30,
                                boxstyle="round,pad=.01,rounding_size=.012",
                                facecolor=PALE, edgecolor=MID, linewidth=.8)
    ax.add_patch(explanation)
    ax.text(.425, .875, "Rows: outer; columns: validation", fontsize=8.0,
            color=INK, va="top")
    ax.text(.425, .815, "Other four groups train in each fit.", fontsize=8.0,
            color=INK, va="top")
    ax.text(.425, .755, "30 ordered cells share 15 pairwise fits.", fontsize=8.0,
            color=INK, va="top")
    ax.text(.425, .695, "Counts apply to each configuration and seed.", fontsize=8.0,
            color=INK, va="top")
    ax.text(.425, .635, "Each family selects its own candidate.", fontsize=8.0,
            color=INK, va="top")

    ax.plot([.035, .965], [.545, .545], color=MID, linewidth=.8)
    ax.text(.035, .515, "(b)", fontsize=10, weight="bold", color=INK, va="top")
    ax.text(.075, .518, "Calibration-dose timeline", fontsize=8.7,
            weight="bold", color=INK, va="top")
    ax.text(.965, .518, "20-ms bins; widths schematic", fontsize=7.8,
            color=MUTED, va="top", ha="right")

    x_start, x_allow, x_gap, x_end = .29, .62, .70, .95
    ax.text((x_start + x_allow) / 2, .462, "label allowance", ha="center",
            va="center", fontsize=8.0, color=CARDINAL_DARK)
    ax.text((x_allow + x_gap) / 2, .462, "gap", ha="center", va="center",
            fontsize=8.0, color=INK)
    ax.text((x_gap + x_end) / 2, .462, "common scored suffix", ha="center",
            va="center", fontsize=8.0, color=SLATE)
    ax.text(x_start, .433, "0", ha="center", va="center", fontsize=7.8, color=MUTED)
    ax.text(x_allow - .008, .433, "3,000", ha="right", va="center",
            fontsize=7.8, color=MUTED)
    ax.text(x_gap + .008, .433, "3,256", ha="left", va="center",
            fontsize=7.8, color=MUTED)

    row_y = (.385, .325, .265, .205)
    row_h = .038
    doses = (("0 s  ·  0 bins", 0), ("15 s  ·  750 bins", 750),
             ("30 s  ·  1,500 bins", 1500), ("60 s  ·  3,000 bins", 3000))
    for (label, bins), y in zip(doses, row_y):
        ax.text(.277, y + row_h / 2, label, ha="right", va="center",
                fontsize=7.8, color=INK)
        ax.add_patch(Rectangle((x_start, y), x_allow - x_start, row_h,
                               facecolor="#F9EDEF", edgecolor=CARDINAL,
                               linewidth=.5))
        labeled_end = x_start + (x_allow - x_start) * bins / 3000
        if bins:
            ax.add_patch(Rectangle((x_start, y), labeled_end - x_start, row_h,
                                   facecolor=CARDINAL, edgecolor=CARDINAL_DARK,
                                   linewidth=.4))
        ax.add_patch(Rectangle((x_allow, y), x_gap - x_allow, row_h,
                               facecolor="#FFF7DF", edgecolor=GOLD,
                               linewidth=.5))
        ax.add_patch(Rectangle((x_gap, y), x_end - x_gap, row_h,
                               facecolor="#E9ECEF", edgecolor=SLATE,
                               linewidth=.5))

    reset_x = x_gap + .56 * (x_end - x_gap)
    ax.plot([reset_x, reset_x], [.185, .425], color=INK, linewidth=.9,
            linestyle=(0, (3, 2)))
    ax.text(reset_x + .008, .425, "recording reset", ha="left", va="bottom",
            fontsize=7.8, color=INK,
            bbox={"facecolor": PAPER, "edgecolor": "none", "pad": .6})
    ax.text(.12, .135,
            "Dashed line: example file boundary; state resets at every recording.",
            fontsize=7.8, color=INK, va="center")
    ax.text(.12, .101,
            "Later files score without a new prefix or gap.",
            fontsize=7.8, color=INK, va="center")
    ax.text(.12, .067,
            "One allowance + one 256-bin gap per concatenated group; scored target labels never enter model updates.",
            fontsize=7.8, color=MUTED, va="center")
    _save(fig, "fig2_protocol")


def load_summary(path: Path) -> dict[str, Any]:
    """Load a complete current result summary or fail without fallback."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Current result summary is absent: {path}. Empirical figures are not generated."
        )
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(f"Result summary is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(summary, dict):
        raise ValueError("summary.json must contain a JSON object")
    if summary.get("schema") != 1:
        raise ValueError("summary.json requires schema=1")
    if summary.get("complete") is not True:
        raise ValueError("summary.json complete must be true; incomplete runs are not plotted")
    if summary.get("study_id") != "h1_channel_temporal_hdc":
        raise ValueError("summary.json study_id must identify the current H1 channel/temporal HDC study")
    if not isinstance(summary.get("protocol_sha256"), str) or not re.fullmatch(
            r"[0-9a-f]{64}", summary["protocol_sha256"]):
        raise ValueError("summary.json requires a 64-character protocol_sha256")
    protocol_path = path.resolve().parent.parent / "provenance" / "protocol.json"
    if not protocol_path.is_file():
        raise FileNotFoundError(f"Current protocol provenance is absent: {protocol_path}")
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    if digest != summary["protocol_sha256"]:
        raise ValueError("summary protocol_sha256 does not match current provenance/protocol.json")
    counts = summary.get("counts")
    if not isinstance(counts, dict) or not counts:
        raise ValueError("summary.json needs completed/expected counts for every experiment stage")
    for stage, count in counts.items():
        if (not isinstance(count, dict) or not isinstance(count.get("completed"), int)
                or not isinstance(count.get("expected"), int) or count["expected"] < 1
                or count["completed"] != count["expected"]):
            raise ValueError(f"experiment stage {stage!r} is incomplete")
    expected_fits = _read_json_object(protocol_path).get("expected_fits")
    if not isinstance(expected_fits, dict) or not expected_fits:
        raise ValueError("current protocol.json lacks expected_fits")
    for stage, expected in expected_fits.items():
        count = counts.get(stage)
        if (not isinstance(expected, int) or not isinstance(count, dict)
                or count.get("expected") != expected or count.get("completed") != expected):
            raise ValueError(f"summary count for {stage!r} does not match frozen protocol")
    session_ids = summary.get("sessions")
    if not isinstance(session_ids, list) or len(session_ids) != 6 or len(set(session_ids)) != 6:
        raise ValueError("complete summary must list six unique session IDs")
    if summary.get("primary_regime") != "few_shot":
        raise ValueError("primary_regime must be few_shot")

    scores = summary.get("scores")
    if not isinstance(scores, list) or not scores:
        raise ValueError("summary.json needs score rows")
    seen_scores, few_shot_families = set(), set()
    for row in scores:
        if not isinstance(row, dict):
            raise ValueError("score rows must be objects")
        family, regime = row.get("family"), row.get("regime")
        if not isinstance(family, str) or regime not in ("few_shot", "zero_shot"):
            raise ValueError("each score row needs a family and valid regime")
        identity = (family, regime, row.get("label", row.get("variant", "")))
        if identity in seen_scores and family != "ablation":
            raise ValueError(f"duplicate score row for {family}/{regime}")
        seen_scores.add(identity)
        if not all(_finite_number(row.get(name)) for name in ("mean", "session_sd", "seed_sd")):
            raise ValueError(f"score row {family}/{regime} has missing/nonfinite statistics")
        _validate_session_rows(row.get("per_session"), session_ids,
                               f"score {family}/{regime}", "mean")
        for session in row["per_session"]:
            seed_scores = session.get("seed_scores")
            if (not isinstance(seed_scores, list) or len(seed_scores) != 10
                    or not all(_finite_number(value) for value in seed_scores)):
                raise ValueError(f"score {family}/{regime} needs ten finite paired-seed scores per session")
        if regime == "few_shot":
            few_shot_families.add(family)
    required = {"hdc", "random", "mlp", "plain", "selected_control"}
    if not required.issubset(few_shot_families):
        raise ValueError(f"few-shot results missing required families: {sorted(required - few_shot_families)}")

    contrasts = summary.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        raise ValueError("summary.json needs paired contrast rows")
    seen_contrasts = set()
    for row in contrasts:
        if not isinstance(row, dict) or not isinstance(row.get("comparator"), str):
            raise ValueError("contrast row needs a comparator")
        regime = row.get("regime")
        if regime not in ("few_shot", "zero_shot"):
            raise ValueError("contrast row needs a valid regime")
        key = (row["comparator"], regime)
        if key in seen_contrasts:
            raise ValueError(f"duplicate contrast for {key}")
        seen_contrasts.add(key)
        if not _finite_number(row.get("mean_difference")):
            raise ValueError(f"contrast {key} needs finite mean_difference")
        _validate_ci(row.get("ci95"), f"contrast {key}")
        _validate_session_rows(row.get("per_session"), session_ids,
                               f"contrast {key}", "delta")
    expected_contrasts = {(family, regime) for family in
                          ("random", "mlp", "plain", "selected_control")
                          for regime in ("few_shot", "zero_shot")}
    if seen_contrasts != expected_contrasts:
        raise ValueError("paired contrasts must include each required comparator in both regimes")

    calibration = summary.get("calibration")
    if not isinstance(calibration, list) or not calibration:
        raise ValueError("summary.json needs calibration measurements")
    _validate_measurement_table(calibration, session_ids, "calibration")
    calibration_families = {row["family"] for row in calibration}
    if calibration_families != required:
        raise ValueError("calibration measurements must cover HDC, all baselines, and inner-selected control")
    for family in calibration_families:
        doses = {row["prefix_seconds"] for row in calibration if row["family"] == family}
        if doses != {0, 15, 30, 60}:
            raise ValueError(f"calibration family {family!r} is missing a prescribed prefix dose")
    robustness = summary.get("robustness")
    if not isinstance(robustness, list) or not robustness:
        raise ValueError("summary.json needs robustness measurements")
    _validate_measurement_table(robustness, session_ids, "robustness")
    robustness_families = {row["family"] for row in robustness}
    if robustness_families != required:
        raise ValueError("robustness measurements must cover HDC, all baselines, and inner-selected control")
    for family in robustness_families:
        for perturbation, levels in (("gain", {0, .1, .2, .4}),
                                     ("dropout", {0, .1, .2, .4})):
            found = {row["level"] for row in robustness
                     if row["family"] == family and row["perturbation"] == perturbation}
            if found != levels:
                raise ValueError(f"robustness {family}/{perturbation} is missing prescribed levels")
    return summary


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON object {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _validate_ci(value: Any, label: str) -> None:
    if (not isinstance(value, list) or len(value) != 2
            or not all(_finite_number(x) for x in value) or value[0] > value[1]):
        raise ValueError(f"{label} needs a finite ordered 95% interval")


def _validate_session_rows(rows: Any, session_ids: list[str], label: str, value_key: str) -> None:
    if not isinstance(rows, list) or len(rows) != len(session_ids):
        raise ValueError(f"{label} must contain one row for each session")
    mapping = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("session"), str):
            raise ValueError(f"{label} has malformed per-session record")
        session = row["session"]
        if session in mapping or not _finite_number(row.get(value_key)):
            raise ValueError(f"{label} has duplicate session or missing {value_key}")
        mapping[session] = row
    if set(mapping) != set(session_ids):
        raise ValueError(f"{label} session IDs do not match the six prespecified groups")


def _validate_measurement_table(rows: list[dict[str, Any]], session_ids: list[str], label: str) -> None:
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("family"), str):
            raise ValueError(f"{label} rows need a family")
        if label == "calibration":
            if row.get("prefix_seconds") not in (0, 15, 30, 60):
                raise ValueError("calibration prefix_seconds must be 0, 15, 30, or 60")
            key = (row["family"], row["prefix_seconds"])
        else:
            if row.get("perturbation") not in ("gain", "dropout") or not _finite_number(row.get("level")):
                raise ValueError("robustness rows need gain/dropout and a numeric level")
            key = (row["family"], row["perturbation"], row["level"])
        if key in seen:
            raise ValueError(f"duplicate {label} row {key}")
        seen.add(key)
        if not _finite_number(row.get("mean")):
            raise ValueError(f"{label} row needs finite mean")
        _validate_ci(row.get("ci95"), label)
        _validate_session_rows(row.get("per_session"), session_ids, label, "mean")


def _score_row(summary: dict[str, Any], family: str, regime: str = "few_shot") -> dict[str, Any]:
    return next(row for row in summary["scores"]
                if row["family"] == family and row["regime"] == regime)


def _uniform_selected_control(summary: dict[str, Any], session_ids: list[str]) -> str | None:
    """Return the selected family only when inner selection agrees in every fold."""
    selection = summary.get("selection")
    if not isinstance(selection, dict):
        return None
    families = []
    for session in session_ids:
        row = selection.get(session)
        family = row.get("selected_control") if isinstance(row, dict) else None
        if family not in ("random", "mlp"):
            return None
        families.append(family)
    return families[0] if len(set(families)) == 1 else None


def _session_value_map(row: dict[str, Any], session_ids: list[str], value_key: str,
                       label: str) -> dict[str, dict[str, Any]]:
    records = row.get("per_session")
    if not isinstance(records, list):
        raise ValueError(f"{label} needs per-session records")
    mapping = {entry.get("session"): entry for entry in records
               if isinstance(entry, dict) and isinstance(entry.get("session"), str)}
    if len(mapping) != len(records) or set(mapping) != set(session_ids):
        raise ValueError(f"{label} session IDs do not match the figure")
    if any(value_key not in entry for entry in mapping.values()):
        raise ValueError(f"{label} per-session records need {value_key}")
    return mapping


def _assert_score_duplicate(summary: dict[str, Any], family: str,
                            session_ids: list[str]) -> None:
    selected = _score_row(summary, family)
    control = _score_row(summary, "selected_control")
    selected_sessions = _session_value_map(selected, session_ids, "mean",
                                           f"{family} score")
    control_sessions = _session_value_map(control, session_ids, "mean",
                                          "selected-control score")
    for session in session_ids:
        left, right = selected_sessions[session], control_sessions[session]
        if left["mean"] != right["mean"] or left.get("seed_scores") != right.get("seed_scores"):
            raise ValueError("Uniform selected-control scores differ from the selected family")
    for field in ("mean", "session_sd", "seed_sd", "ci95"):
        if selected.get(field) != control.get(field):
            raise ValueError("Uniform selected-control score summary differs from the selected family")

    contrasts = {(row.get("comparator"), row.get("regime")): row
                 for row in summary.get("contrasts", [])}
    primary = contrasts.get((family, "few_shot"))
    selected_control = contrasts.get(("selected_control", "few_shot"))
    if primary is None or selected_control is None:
        raise ValueError("Primary and family contrast rows are required for Fig. 3")
    primary_sessions = _session_value_map(primary, session_ids, "delta",
                                          f"{family} contrast")
    control_sessions = _session_value_map(selected_control, session_ids, "delta",
                                          "selected-control contrast")
    if any(primary_sessions[s]["delta"] != control_sessions[s]["delta"]
           for s in session_ids):
        raise ValueError("Uniform selected-control contrasts differ from the selected family")
    for field in ("mean_difference", "ci95", "positive_groups",
                  "exploratory_sign_flip_p"):
        if field in primary or field in selected_control:
            if primary.get(field) != selected_control.get(field):
                raise ValueError("Uniform selected-control contrast summary differs from the selected family")


def _assert_measurement_curve_duplicate(summary: dict[str, Any], family: str,
                                        session_ids: list[str], table_name: str) -> None:
    rows = summary.get(table_name)
    if not isinstance(rows, list):
        raise ValueError(f"{table_name} measurements are required for Fig. 4")
    def identity(row: dict[str, Any]) -> tuple[Any, ...]:
        if table_name == "calibration":
            return (row.get("prefix_seconds"),)
        return (row.get("perturbation"), row.get("level"))

    selected = {identity(row): row for row in rows if row.get("family") == family}
    control = {identity(row): row for row in rows if row.get("family") == "selected_control"}
    if not selected or set(selected) != set(control):
        raise ValueError(f"Uniform selected-control {table_name} curve is incomplete")
    for key in selected:
        left, right = selected[key], control[key]
        for field in ("mean", "session_sd", "seed_sd", "ci95"):
            if left.get(field) != right.get(field):
                raise ValueError(f"Uniform selected-control {table_name} summary differs from {family}")
        left_sessions = _session_value_map(left, session_ids, "mean",
                                           f"{family} {table_name} measurement")
        right_sessions = _session_value_map(right, session_ids, "mean",
                                            f"selected-control {table_name} measurement")
        for session in session_ids:
            a, b = left_sessions[session], right_sessions[session]
            if a["mean"] != b["mean"] or a.get("seed_scores") != b.get("seed_scores"):
                raise ValueError(f"Uniform selected-control {table_name} curve differs from {family}")


def figure3_session_results(summary: dict[str, Any]) -> None:
    """Actual per-session R2 and paired family deltas, with duplicate controls merged."""
    ids = summary["sessions"]
    selected_family = _uniform_selected_control(summary, ids)
    if selected_family is not None:
        _assert_score_duplicate(summary, selected_family, ids)
    families = ["hdc", "random", "mlp", "plain"]
    if selected_family is None:
        families.append("selected_control")
    fig, axes = plt.subplots(1, 2, figsize=(174 / 25.4, 104 / 25.4),
                             gridspec_kw={"width_ratios": [1.45, 1]})
    ax = axes[0]
    x = np.arange(len(ids))
    offsets = dict(zip(families, np.linspace(-.32, .32, len(families))))
    for family in families:
        row = _score_row(summary, family)
        by_session = {entry["session"]: entry["mean"] for entry in row["per_session"]}
        vals = np.asarray([by_session[sid] for sid in ids], dtype=float)
        label = (f"{'MLP' if family == 'mlp' else 'Random'} (selected control)"
                 if family == selected_family else FAMILY_LABELS[family])
        ax.scatter(x + offsets[family], vals, s=34, color=COLORS[family],
                   marker=MARKERS[family], label=label,
                   edgecolor=PAPER, linewidth=.45, zorder=3)
    ax.set_xticks(x, ids)
    ax.set_xlim(-.55, len(ids) - .45)
    ax.set_ylabel(r"Calibrated group $R^2$")
    ax.set_xlabel("Outer test group")
    ax.grid(axis="y", color=MID, linewidth=.55)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(a) Outer-group means", loc="left", fontsize=8.5, weight="bold")

    ax = axes[1]
    contrast_order = ["random", "mlp", "plain"]
    if selected_family is None:
        contrast_order.append("selected_control")
    contrast_by_family = {row["comparator"]: row for row in summary["contrasts"]
                          if row["regime"] == "few_shot"}
    contrast_rows = [contrast_by_family[family] for family in contrast_order]
    delta_families = [row["comparator"] for row in contrast_rows]
    positions = np.arange(len(delta_families))
    ax.axvline(0, color=INK, lw=.8)
    for pos, row in zip(positions, contrast_rows):
        family = row["comparator"]
        color = COLORS.get(family, SLATE)
        by_session = {entry["session"]: entry["delta"] for entry in row["per_session"]}
        delta = np.asarray([by_session[sid] for sid in ids], dtype=float)
        jitter = np.linspace(-.14, .14, len(ids))
        ax.scatter(delta, pos + jitter, color=color,
                   marker=MARKERS.get(family, "o"), s=27,
                   edgecolor=PAPER, linewidth=.35, zorder=3)
        mean = row["mean_difference"]
        lo, hi = row["ci95"]
        ax.errorbar([mean], [pos], xerr=[[mean - lo], [hi - mean]], fmt="none",
                    color=color, lw=1.25, capsize=2.5, zorder=2)
        ax.plot([0, mean], [pos, pos], color=color, lw=2.1,
                solid_capstyle="round", zorder=2)
        ax.scatter([mean], [pos], marker="|", s=110, color=INK, linewidth=1.1,
                   zorder=4)
    short_labels = {"random": "Random", "mlp": "MLP", "plain": "GRU",
                    "selected_control": "Selected control"}
    primary_family = selected_family if selected_family is not None else "selected_control"
    tick_labels = []
    for family in delta_families:
        label = short_labels[family]
        if family == primary_family:
            label = f"{label}\n(primary)" if family == "selected_control" else f"{label} (primary)"
        tick_labels.append(label)
    ax.set_yticks(positions, tick_labels)
    for family, tick in zip(delta_families, ax.get_yticklabels()):
        if family == primary_family:
            tick.set_fontweight("bold")
    ax.invert_yaxis()
    ax.set_xlabel(r"$\Delta R^2$: HDC minus comparator")
    ax.set_title("(b) Paired contrasts", loc="left", fontsize=8.5, weight="bold")
    ax.grid(axis="x", color=MID, linewidth=.55)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.subplots_adjust(left=.10, right=.985, bottom=.29, top=.88, wspace=.50)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, fontsize=8.0,
               loc="lower center", bbox_to_anchor=(.52, .02),
               columnspacing=1.3, handletextpad=.45)
    _save(fig, "fig3_session_results")


def figure4_adaptation_robustness(summary: dict[str, Any]) -> None:
    """Actual calibration-budget and input-corruption measurements."""
    fig, axes = plt.subplots(1, 3, figsize=(174 / 25.4, 111 / 25.4), sharey=True)
    calibration = summary["calibration"]
    ids = summary["sessions"]
    selected_family = _uniform_selected_control(summary, ids)
    if selected_family is not None:
        _assert_measurement_curve_duplicate(summary, selected_family, ids, "calibration")
        _assert_measurement_curve_duplicate(summary, selected_family, ids, "robustness")
    families = ["hdc", "random", "mlp", "plain"]
    if selected_family is None:
        families.append("selected_control")
    titles = (("Calibration budget", "prefix_seconds", "Calibration prefix (s)"),
              ("Channel gain shift", "gain", "Gain log-SD"),
              ("Channel dropout", "dropout", "Fraction dropped"))
    legend_handles = []
    for index, (ax, (title, measure, xlabel)) in enumerate(zip(axes, titles)):
        if index == 0:
            rows_by_family = {
                family: sorted((row for row in calibration if row["family"] == family),
                               key=lambda row: row["prefix_seconds"])
                for family in families
            }
            x_key = "prefix_seconds"
        else:
            perturbation = measure
            robustness = [row for row in summary["robustness"]
                          if row["perturbation"] == perturbation]
            rows_by_family = {
                family: sorted((row for row in robustness if row["family"] == family),
                               key=lambda row: row["level"])
                for family in families
            }
            x_key = "level"
        for family in families:
            rows = rows_by_family[family]
            xx = np.asarray([row[x_key] for row in rows], dtype=float)
            yy = np.asarray([row["mean"] for row in rows], dtype=float)
            lo = np.asarray([row["ci95"][0] for row in rows], dtype=float)
            hi = np.asarray([row["ci95"][1] for row in rows], dtype=float)
            color = COLORS[family]
            label = (f"{'MLP' if family == 'mlp' else 'Random'} (selected control)"
                     if family == selected_family else FAMILY_LABELS[family])
            line, = ax.plot(xx, yy, color=color, marker=MARKERS[family],
                            markersize=4.2, lw=1.25, label=label,
                            zorder=3)
            ax.errorbar(xx, yy, yerr=np.vstack((yy - lo, hi - yy)), fmt="none",
                        ecolor=color, elinewidth=.9, capsize=2.2, zorder=2)
            if index == 0:
                legend_handles.append(line)
        ax.set_xlabel(xlabel)
        ax.set_xticks([0, 15, 30, 60] if index == 0 else [0, .1, .2, .4])
        ax.set_title(f"({'abc'[index]}) {title}", loc="left", fontsize=8.5, weight="bold")
        ax.grid(color=MID, linewidth=.5)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="both", labelsize=8)
    axes[0].set_ylabel(r"Mean group $R^2$")
    for ax in axes[1:]:
        ax.tick_params(labelleft=False)
    fig.subplots_adjust(left=.095, right=.985, bottom=.34, top=.87, wspace=.34)
    fig.text(.5, .205,
             "All calibration doses share target bins; 0 s uses no session labels.\n"
             "Gain and dropout panels use zero-label decoding.",
             ha="center", va="center", fontsize=7.2, color=INK, linespacing=1.3)
    fig.legend(legend_handles,
               [f"{'MLP' if f == 'mlp' else 'Random'} (selected control)"
                if f == selected_family else FAMILY_LABELS[f] for f in families],
               frameon=False, ncol=3, fontsize=8.0,
               loc="lower center", bbox_to_anchor=(.52, .02),
               columnspacing=1.2, handletextpad=.45)
    _save(fig, "fig4_calibration_robustness")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=SUMMARY_DEFAULT,
                        help="current complete result summary; no legacy fallback")
    parser.add_argument("--conceptual-only", action="store_true",
                        help="generate mechanism/protocol diagrams without empirical results")
    args = parser.parse_args(argv)

    figure1_mechanism()
    figure2_protocol()
    if args.conceptual_only:
        print(f"Generated conceptual figures in {FIG_DIR}")
        return 0
    summary = load_summary(args.summary)
    figure3_session_results(summary)
    figure4_adaptation_robustness(summary)
    print(f"Generated Figures 1 through 4 in {FIG_DIR}")
    if (ROOT.parent / "neuroinformatics_submission/article.tex").is_file():
        if __package__:
            from .build_submission import build
        else:
            from build_submission import build
        build()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
