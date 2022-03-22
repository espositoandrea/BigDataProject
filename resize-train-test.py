#!/usr/bin/env python3
import argparse
import shutil
from collections import namedtuple
from functools import partial
from pathlib import Path
import random
from typing import List, Tuple

import pandas as pd
import yaml
from p_tqdm import p_umap


def setup_args() -> argparse.Namespace:
    if Path("params.yaml").exists():
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    else:
        params = {'resize': {'to-drop': ["monitoredCall/StopPointName", "monitoredCall/StopPointRef"]}}
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing the windows')
    parser.add_argument('output', help="Output directory")
    parser.add_argument('--drop', '-d', action='append')
    args = parser.parse_args()
    if not args.drop:
        args.drop = params['resize']['to-drop']
    args.drop = tuple(args.drop)
    return args


def fix_dataset(file: Path, output: Path, drop: Tuple[str] = ()):
    reader = pd.read_csv(file, usecols=lambda x: not x.startswith(drop), chunksize=1000)
    header = True
    for chunk in reader:
        chunk.to_csv(output / file.name, index=True, index_label='id', header=header, mode='a')
        header = False


def fix_split(outdir: Path, split: Path, drop: Tuple[str] = ()):
    output = outdir / split.absolute().name
    output.mkdir(exist_ok=True, parents=True)
    fix_dataset(split / 'train.csv', output, drop=drop)
    shutil.copy(split / 'perturbations.csv', output)
    fix_dataset(split / 'test.csv', output, drop=drop)


def main():
    args = setup_args()
    directory = Path(args.directory)
    output = Path(args.output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(exist_ok=True, parents=True)
    files = list(directory.glob('split-*'))
    func = partial(fix_split, output, drop=args.drop)
    p_umap(func, files)


if __name__ == '__main__':
    main()
