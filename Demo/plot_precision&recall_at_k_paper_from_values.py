"""
Paper-grade Precision@K bar chart (grouped).

Uses hard-coded values (provided by user) to avoid CSV dependency.

Output:
  Demo/ablation_precision_K_paper.png
"""

import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # 1) Fixed order (user requested)
    ablation_order = ["MechaGRN", "w/o E", "w/o H", "w/o D", "w/o M"]
    k_values = [50, 100, 200]
    
    # 2) User-provided Precision@K values
    precision = {
        "MechaGRN": {50: 0.96, 100: 0.93, 200: 0.89},
        "w/o E": {50: 0.94, 100: 0.91, 200: 0.865},
        "w/o H": {50: 0.90, 100: 0.85, 200: 0.78},
        "w/o D": {50: 0.44, 100: 0.38, 200: 0.32},
        "w/o M": {50: 0.26, 100: 0.21, 200: 0.18},
    }

    recall = {
        "MechaGRN": {50: 0.027429, 100: 0.053143, 200: 0.101714},
        "w/o E": {50: 0.026857, 100: 0.052, 200: 0.098857},
        "w/o H": {50: 0.025714, 100: 0.048571, 200: 0.089143},
        "w/o D": {50: 0.012571, 100: 0.021714, 200: 0.036571},
        "w/o M": {50: 0.007429, 100: 0.012, 200: 0.020571},
    }
    index = "Recall" # "Precision"
    datas = precision if index == "Precision" else recall
    # 3) Paper-friendly colors (浅到深 by K)
    # Light -> Medium -> Dark
    colors = ["#A6CEE3", "#1F78B4", "#08306B"]  # 3 bars

    values = np.array([[datas[abl][k] for k in k_values] for abl in ablation_order], dtype=float)
    # values: shape (num_ablation, num_k)

    x = np.arange(len(ablation_order))
    bar_width = 0.22

    fig, ax = plt.subplots(figsize=(9.5, 5.3))
    plt.rcParams['font.weight'] = 'bold'
    num_k = len(k_values)
    offsets = [(j - (num_k - 1) / 2.0) * bar_width for j in range(num_k)]

    for j, K in enumerate(k_values):
        bar_x = x + offsets[j]
        bars = ax.bar(
            bar_x,
            values[:, j],
            width=bar_width,
            color=colors[j % len(colors)],
            edgecolor="black",
            linewidth=0.4,
            label=f"K={K}",
        )

        # 4) Annotate values above each bar (3 decimals)
        for b in bars:
            h = float(b.get_height())
            # Keep text inside plot area near top
            if h >= 0.99:
                y_text = h - 0.04 if index == "Precision" else h - 0.002
                va = "top"
            else:
                y_text = h + 0.02 if index == "Precision" else h + 0.002
                va = "bottom"
            ax.text(
                b.get_x() + b.get_width() / 2.0,
                y_text,
                f"{h:.3f}",
                ha="center",
                va=va,
                fontsize=9,
            )

    # 5) Axis / grid / title
    ax.set_title(f"{index}@K Comparison in Abalation Study", fontsize=21, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(ablation_order, fontsize=16)
    ax.set_ylim(0.0, 1.05) if index == "Precision" else ax.set_ylim(0.0, 0.11)
    ax.set_ylabel(f"{index}@K", fontsize=21, fontweight='bold')
    
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)

    # Optional: ensure readable y ticks
    ax.set_yticks(np.linspace(0, 1, 6)) if index == "Precision" else ax.set_yticks(np.linspace(0, 0.11, 6))

    ax.legend(frameon=True, fontsize=11)
    fig.tight_layout()

    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, f"ablation_{index}_K_paper.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[INFO] Saved: {out_path}")


if __name__ == "__main__":
    main()

