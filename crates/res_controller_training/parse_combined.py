import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(ds_id, df):
    # mean mse for each (buffer_size, pca)
    pivot = (
        df[df["dataset_id"] == ds_id]
        .groupby(["buffer_size", "pca"])["mse"]
        .mean()
        .unstack("pca")
        .sort_index()
    )

    buffers = pivot.index.to_list()
    pcas = pivot.columns.to_list()
    grid = pivot.values

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(grid, aspect="auto")

    ax.set_title(f"Imitation MSE heatmap (dataset={ds_id})")
    ax.set_xlabel("PCA dimension k")
    ax.set_ylabel("Buffer size B")

    ax.set_xticks(range(len(pcas)))
    ax.set_xticklabels([str(k) for k in pcas])
    ax.set_yticks(range(len(buffers)))
    ax.set_yticklabels([str(b) for b in buffers])

    # annotate each cell with the mean value
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.4f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean MSE (lower is better)")

    fig.tight_layout()
    fig.savefig(f"heatmap_{ds_id}.pdf", dpi=200)
    plt.close(fig)


def main():
    # dataset_id, simulation_id, buffer_size, pca, mse
    df = pd.read_csv(
        "combined.csv",
        header=None,
        names=["dataset_id", "simulation_id", "buffer_size", "pca", "mse"],
    )

    for ds_id in sorted(df["dataset_id"].unique()):
        plot_heatmap(ds_id, df)


if __name__ == "__main__":
    main()
