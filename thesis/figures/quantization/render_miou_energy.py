"""Render the thesis quantization plot directly from its test-results table.

The logarithmic energy axis preserves readability with the new FP32 outlier.
No validation accuracy is substituted for the existing test measurements.
The default destination is a QA preview, not the thesis image itself.
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE = (
    ROOT / "thesis/latex/Chapters"
    / "5 Evaluation of Lightweight Scene Understanding.tex"
)
MODELS = {
    "U-Net (ResNet34)": ("U-Net / R34", "#0072B2"),
    "U-Net (MobileNetV2)": ("U-Net / MV2", "#009E73"),
    "DeepLabV3 (ResNet34)": ("DeepLabV3 / R34", "#D55E00"),
    "DeepLabV3+ (MobileNetV2)": ("DeepLabV3+ / MV2", "#CC79A7"),
    "SegFormer-B0": ("SegFormer-B0", "#E69F00"),
}
MARKERS = {"FP32": "o", "FP16": "s", "PTQ INT8": "^", "QAT INT8": "D"}


def read_points(source):
    text = source.read_text(encoding="utf-8-sig")
    label = text.index(r"\label{tab:s5mars-quantization-accuracy-efficiency}")
    start = text.rfind(r"\begin{table}", 0, label)
    end = text.index(r"\end{table}", label)
    model = None
    points = []
    for line in text[start:end].splitlines():
        heading = re.search(r"\\textit\{([^}]+)\}", line)
        if heading:
            model = heading.group(1)
        if re.match(r"\s*(FP32|FP16|PTQ INT8|QAT INT8)\s*&", line):
            fields = line.split("&")
            if len(fields) != 6 or model not in MODELS:
                raise ValueError(f"Unexpected quantization row: {line}")
            points.append({
                "model": model,
                "variant": fields[0].strip(),
                "test_miou_percent": float(fields[1]),
                "energy_mj_per_image": float(fields[4]),
            })
    expected = {(model, variant) for model in MODELS for variant in MARKERS}
    actual = {(point["model"], point["variant"]) for point in points}
    if len(points) != 20 or actual != expected:
        raise ValueError("Expected one measurement for each of the 20 configurations")
    if any(not (0 <= p["test_miou_percent"] <= 100)
           or p["energy_mj_per_image"] <= 0 for p in points):
        raise ValueError("Invalid accuracy or energy measurement")
    return points


def pareto_frontier(points):
    return [p for p in points if not any(
        q["energy_mj_per_image"] <= p["energy_mj_per_image"]
        and q["test_miou_percent"] >= p["test_miou_percent"]
        and (q["energy_mj_per_image"] < p["energy_mj_per_image"]
             or q["test_miou_percent"] > p["test_miou_percent"])
        for q in points
    )]


def render(points, output):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.linewidth": 1.1,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
    })
    fig, ax = plt.subplots(figsize=(8.8, 6.25))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.985, bottom=0.30)
    ax.set_axisbelow(True)
    ax.set_xscale("log")
    ax.set_xlim(100, 4500)
    ax.set_ylim(57.5, 79)
    ax.xaxis.set_major_locator(FixedLocator([100, 200, 400, 800, 1600, 3200]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.0f}"))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks([60, 63, 66, 69, 72, 75, 78])
    ax.grid(color="#dedede", linewidth=0.8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for point in points:
        ax.scatter(
            point["energy_mj_per_image"], point["test_miou_percent"],
            marker=MARKERS[point["variant"]], s=80,
            facecolor=MODELS[point["model"]][1], edgecolor="#303030",
            linewidth=1, zorder=3,
        )
    ax.set_xlabel("GPU energy per image (mJ, logarithmic scale)", labelpad=9)
    ax.set_ylabel("Test mIoU (%)", labelpad=8)
    lookup = {(p["model"], p["variant"]): p for p in points}
    annotations = [
        ("SegFormer-B0", "QAT INT8", "SegFormer QAT", (470, 77.5)),
        ("DeepLabV3+ (MobileNetV2)", "QAT INT8", "DeepLabV3+ QAT", (265, 68.1)),
        ("DeepLabV3 (ResNet34)", "FP32", "DeepLabV3 FP32", (1700, 71.1)),
    ]
    for model, variant, label, position in annotations:
        point = lookup[(model, variant)]
        ax.annotate(
            label, (point["energy_mj_per_image"], point["test_miou_percent"]),
            xytext=position, ha="left", va="center", fontsize=12,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 1},
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )
    model_handles = [Line2D(
        [], [], marker="o", linestyle="none", markersize=8,
        markerfacecolor=color, markeredgecolor="#303030", label=label,
    ) for label, color in MODELS.values()]
    precision_handles = [Line2D(
        [], [], marker=marker, linestyle="none", markersize=8,
        markerfacecolor="white", markeredgecolor="#303030", label=variant,
    ) for variant, marker in MARKERS.items()]
    fig.legend(
        handles=model_handles, loc="center", bbox_to_anchor=(0.54, 0.16),
        ncol=3, frameon=False, columnspacing=1.8, handletextpad=0.6,
    )
    fig.legend(
        handles=precision_handles, loc="center", bbox_to_anchor=(0.54, 0.047),
        ncol=4, frameon=False, columnspacing=2, handletextpad=0.6,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=240, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).with_name("quantization_miou_energy_updated.png"),
    )
    args = parser.parse_args()
    points = read_points(args.source)
    render(points, args.output)
    print(json.dumps({
        "output": str(args.output), "points": len(points),
        "deeplabv3": [p for p in points if p["model"] == "DeepLabV3 (ResNet34)"],
        "non_dominated": pareto_frontier(points),
    }, indent=2))
