#!/usr/bin/env python3

import argparse
from functools import partial

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from pathlib import Path
from p_tqdm import p_umap


def test_with_k_clusters(df: pd.DataFrame, k: int, seed: int = 42, outfolder: str = 'results'):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, df.shape[0] + (k + 1) * 10])

    clusterer = KMeans(n_clusters=k, random_state=seed)
    labels = clusterer.fit_predict(df)
    silhouette_avg = silhouette_score(df, labels)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(df, labels)
    y_lower = 10
    for i in range(k):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / k)
        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            facecolor=color,
            edgecolor=color,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(labels.astype(float) / k)
    ax2.scatter(
        df.iloc[:, 0], df.iloc[:, 1], marker="o", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
    )

    # Labeling the clusters
    centers = clusterer.cluster_centers_
    # Draw white circles at cluster centers
    ax2.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="o",
        c="white",
        alpha=1,
        s=200,
        edgecolor="k",
    )

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(
        "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
        % k,
        fontsize=14,
        fontweight="bold",
    )

    directory = Path(outfolder)
    directory.mkdir(parents=True, exist_ok=True)
    plt.savefig(directory / f"{k:02d}-clusters.pdf")
    plt.clf()
    return k, silhouette_avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='+')
    parser.add_argument('--seed', '-s', default=42, type=int)
    parser.add_argument('--min', '-m', default=2, type=int)
    parser.add_argument('--max', '-M', default=20, type=int)
    args = parser.parse_args()

    load = partial(pd.read_csv, usecols=["Latitude", "Longitude"])

    df = pd.concat(map(load, args.infile), axis=0, ignore_index=True)
    df = df[(df != 0).any(axis=1)]

    test_k_on_dataset = partial(test_with_k_clusters, df, seed=args.seed)

    res = p_umap(test_k_on_dataset, list(range(args.min, args.max + 1)))
    print("n_clusters,silhouette_score")
    print("\n".join(map(lambda x: f"{x[0]},{x[1]}", sorted(res, key=lambda x: x[0]))))


if __name__ == '__main__':
    main()
