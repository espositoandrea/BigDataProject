#!/usr/bin/env python3
import re
import sys
import argparse

import pandas as pd
import numpy as np


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--seed', '-s', default=42, type=int)
    parser.add_argument('--min', '-m', default=2, type=int)
    parser.add_argument('--max', '-M', default=20, type=int)
    return parser.parse_args()


DROP_COLS = [
    "LinkDistance",
    "LineRef",
    "OriginName",
    "DestinationName",
    "VehicleRef",
    "monitoredCall/DestinationDisplay",
]


def main():
    args = setup_args()
    df = pd.read_csv(args.infile,
                     usecols=lambda x: x not in DROP_COLS,
                     dtype={
                         'HeadwayService': "boolean",
                         "InCongestion": 'boolean',
                         'InPanic': 'boolean',
                         'monitoredCall/VehicleAtStop': 'boolean'
                     },
                     parse_dates=["dateTime"])
    df.drop(labels=62, inplace=True) # This has some Null values: how can we treat it? We should make this code more general
    df["Delay"] = df["Delay"].map(lambda x: int(re.sub(r"(-?)PT(\d+)S", r"\1\2", str(x))))
    indexer = df["dateTime"].sort_values().diff().fillna(pd.Timedelta(seconds=0)).cumsum().astype('int').map(lambda x: int(np.floor(x * 1e-9 / 300)))
    df["dateTimeGroup"] = indexer
    print("---")
    print(indexer)
    grp = df.groupby(['Cluster', indexer])
    #print(df.head())
    print(grp.aggregate({
        'dateTime': 'min',
        'Delay': 'mean',
        'Percentage': 'mean',
        'InPanic': 'sum',
        'InCongestion': 'sum',
    }))
    pass


if __name__ == '__main__':
    main()
