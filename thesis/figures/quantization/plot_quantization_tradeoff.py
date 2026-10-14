"""Plot recorded test accuracy against independently benchmarked GPU energy.

The CSV is a curated transcription of the current local PTQ and ModelOpt QAT
reports. Empty resource fields are genuinely unavailable, not zero. SegFormer
uses the selected Conv-only variants, not the extended quantization ablations.
No model is loaded and no inference or training is performed by this script.
"""

import argparse
import csv
import math
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


MODELS = {
    "U-Net / ResNet34": ("U-Net / R34", "#0072B2"),
    "U-Net / MobileNetV2": ("U-Net / MV2", "#009E73"),
    "DeepLabV3 / ResNet34": ("DeepLabV3 / R34", "#D55E00"),
    "DeepLabV3+ / MobileNetV2": ("DeepLabV3+ / MV2", "#CC79A7"),
    "SegFormer-B0": ("SegFormer-B0", "#E69F00"),
}
VARIANTS = {
    "FP32": ("FP32", "o"),
    "FP16": ("FP16", "s"),
    "PTQ": ("PTQ INT8", "^"),
    "QAT": ("QAT INT8", "D"),
}


def load_results(path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    expected = {(model, variant) for model in MODELS for variant in VARIANTS}
    observed = [(row["model"], row["variant"]) for row in rows]
    if len(observed) != len(set(observed)) or set(observed) != expected:
        raise ValueError("Expected exactly one row per model and variant")
    missing = set()
    for row in rows:
        for field in ("test_miou", "latency_ms", "energy_mj", "gpu_peak_mib"):
            raw = row[field]
            row[field] = float(raw) if raw else None
            value = row[field]
            if value is None:
                missing.add((row["model"], row["variant"], field))
            elif not math.isfinite(value) or value <= 0:
                raise ValueError(f"Invalid {field}: {row}")
        if not 0 < row["test_miou"] <= 1:
            raise ValueError("mIoU must be stored as a fraction")
    expected_missing = {
        ("DeepLabV3 / ResNet34", "FP32", "latency_ms"),
        ("DeepLabV3 / ResNet34", "FP32", "energy_mj"),
        ("DeepLabV3 / ResNet34", "FP32", "gpu_peak_mib"),
        ("DeepLabV3 / ResNet34", "FP16", "energy_mj"),
        ("DeepLabV3 / ResNet34", "FP16", "gpu_peak_mib"),
    }
    if missing != expected_missing:
        raise ValueError(f"Unexpected missing values: {missing}")
    return rows


def report_checks(rows):
    index = {(row["model"], row["variant"]): row for row in rows}
    print("Validated 20 configurations; 18 have measured energy.")
    print("LaTeX rows: model | variant | mIoU (%) | delta (pp) | ms | mJ | MiB")
    fp16_reductions = []
    for row in rows:
        baseline = index[(row["model"], "FP32")]
        delta = 100 * (row["test_miou"] - baseline["test_miou"])
        fields = []
        for field in ("latency_ms", "energy_mj", "gpu_peak_mib"):
            value = row[field]
            fields.append(
                "TBD" if value is None else str(
                    Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
                )
            )
        print(
            f"{row['model']} | {row['variant']} | "
            f"{100 * row['test_miou']:.2f} | {delta:+.2f} | "
            + " | ".join(fields)
        )
        if row["variant"] == "FP16" and row["energy_mj"] is not None:
            fp16_reductions.append(100 * (1 - row["energy_mj"] / baseline["energy_mj"]))
    print(
        "FP16 paired energy reductions: "
        f"{min(fp16_reductions):.1f}% to {max(fp16_reductions):.1f}%"
    )
    measured = [row for row in rows if row["energy_mj"] is not None]
    frontier = [
        row for row in measured
        if not any(
            other["energy_mj"] <= row["energy_mj"]
            and other["test_miou"] >= row["test_miou"]
            and (
                other["energy_mj"] < row["energy_mj"]
                or other["test_miou"] > row["test_miou"]
            )
            for other in measured
        )
    ]
    print("Non-dominated recorded points:", [(r["model"], r["variant"]) for r in frontier])
    return measured


def plot_results(rows, output_path):
    plt.rcParams.update({
        "font.family": "DejaVu Serif",
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "savefig.dpi": 350,
    })
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.965, bottom=0.31)
    ax.set_axisbelow(True)
    ax.grid(color="#D8D8D8", linewidth=0.6, alpha=0.8)
    for row in rows:
        _, color = MODELS[row["model"]]
        _, marker = VARIANTS[row["variant"]]
        ax.scatter(
            row["energy_mj"], 100 * row["test_miou"],
            marker=marker, s=62, c=color,
            edgecolors="#222222", linewidths=0.6, zorder=3,
        )
    ax.set_xlabel("GPU energy per image (mJ)", labelpad=7)
    ax.set_ylabel("Test mIoU (%)", labelpad=7)
    ax.set_xlim(60, 1280)
    ax.set_ylim(57.5, 79)
    ax.set_xticks(range(200, 1201, 200))
    ax.set_yticks(range(60, 79, 3))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotation_positions = {
        "DeepLabV3+ / MobileNetV2": ((360, 68.4), "DeepLabV3+ QAT"),
        "SegFormer-B0": ((585, 77.7), "SegFormer QAT"),
    }
    for model, (position, text) in annotation_positions.items():
        row = next(r for r in rows if r["model"] == model and r["variant"] == "QAT")
        ax.annotate(
            text, xy=(row["energy_mj"], 100 * row["test_miou"]),
            xytext=position, fontsize=9.5,
            arrowprops={"arrowstyle": "-", "color": "#555555", "lw": 0.8},
            ha="left", va="center", zorder=4,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1},
        )
    model_handles = [
        Line2D([], [], linestyle="none", marker="o", markersize=6.5,
               markerfacecolor=color, markeredgecolor="#222222", label=label)
        for label, color in MODELS.values()
    ]
    variant_handles = [
        Line2D([], [], linestyle="none", marker=marker, markersize=6.5,
               markerfacecolor="white", markeredgecolor="#222222", label=label)
        for label, marker in VARIANTS.values()
    ]
    fig.legend(model_handles, [h.get_label() for h in model_handles],
               loc="center", bbox_to_anchor=(0.52, 0.155), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.25, handletextpad=0.4)
    fig.legend(variant_handles, [h.get_label() for h in variant_handles],
               loc="center", bbox_to_anchor=(0.52, 0.045), ncol=4,
               frameon=False, fontsize=9.5, columnspacing=1.7, handletextpad=0.4)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    print(f"Saved chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).with_name("quantization_results.csv"))
    args = parser.parse_args()
    rows = load_results(args.data)
    measured = report_checks(rows)
    plot_results(measured, args.output)


if __name__ == "__main__":
    main()
