#!/usr/bin/env python3
import argparse
import sys
from math import floor
from pathlib import Path

import pandas as pd


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output', '-o', help="Output directory", default='.')
    return parser.parse_args()


def process(chunk: pd.DataFrame, basedir='windows'):
    base = Path(basedir)
    base.mkdir(exist_ok=True, parents=True)
    grouper = chunk['TimeWindowID'].map(lambda x: floor(x / 60))
    for name, grp in chunk.groupby(grouper):
        filename = base / f"window-{name}.csv"
        grp.to_csv(filename, mode='a', header=(not filename.exists()), index=False)


def main():
    args = setup_args()
    with pd.read_csv(args.infile, parse_dates=["dateTimeGroup"], chunksize=100) as reader:
        list(map(lambda x: process(x, basedir=args.output), reader))


if __name__ == '__main__':
    main()
