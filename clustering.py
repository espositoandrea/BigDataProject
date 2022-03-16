#!/usr/bin/env python3

import argparse
import logging
import pickle
import yaml
import sys
from functools import partial
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.cluster import KMeans


def train_model(datasets: List[str], k: int, seed: int = 42, outfile: str = None, force: bool = False) -> pd.DataFrame:
    load = partial(pd.read_csv, dtype={'HeadwayService': "boolean", "InCongestion": 'boolean', 'InPanic': 'boolean',
                                       'monitoredCall/VehicleAtStop': 'boolean'})
    df = pd.concat(map(load, datasets), axis=0, ignore_index=True)
    df = df[(df != 0).any(axis=1)]  # Filter out errors
    if Path(outfile).exists() and not force:
        logging.info("Loading existing model")
        with open(outfile, "rb") as f:
            clusterer = pickle.load(f)
        labels = clusterer.predict(df[["Longitude", "Latitude"]])
    else:
        logging.info("Training new model")
        clusterer = KMeans(n_clusters=k, random_state=seed)
        labels = clusterer.fit_predict(df[["Longitude", "Latitude"]])
        if outfile:
            with open(outfile, "wb") as f:
                pickle.dump(clusterer, f)
    centroids = clusterer.cluster_centers_
    df["Cluster"] = labels
    df[["ClusterLongitude", "ClusterLatitude"]] = centroids[labels]
    return df


def main():
    if Path("params.yaml").exists():
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    else:
        params = {
            'seed': 42,
            'clustering': {
                'k': 100
            }
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='+')
    parser.add_argument('--seed', '-s', default=params['seed'], type=int)
    parser.add_argument('--clusters', '-k', default=params['clustering']['k'], type=int)
    parser.add_argument('--force', '-f', action='store_true')
    parser.add_argument('--model', '-m', default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    labels = train_model(args.infile, args.clusters, args.seed, outfile=args.model, force=args.force)
    labels.to_csv(sys.stdout, index=False)


if __name__ == '__main__':
    main()
