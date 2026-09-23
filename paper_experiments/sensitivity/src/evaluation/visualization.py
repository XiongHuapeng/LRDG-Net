"""LRDG-Net 科研输出可视化。 / Research-result visualization utilities."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def save_confusion_matrix(matrix, csv_path, png_path, title: str) -> None:
    matrix = np.asarray(matrix, dtype=int)
    pd.DataFrame(
        matrix,
        index=["true_clean", "true_contaminated"],
        columns=["pred_clean", "pred_contaminated"],
    ).to_csv(csv_path, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(matrix)
    ax.set_xticks([0, 1], ["clean", "contaminated"])
    ax.set_yticks([0, 1], ["clean", "contaminated"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(Path(png_path), dpi=180)
    plt.close(fig)
