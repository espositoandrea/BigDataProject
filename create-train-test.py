#!/usr/bin/env python3
import argparse
import shutil
from collections import namedtuple
from functools import partial
from pathlib import Path
import random
from typing import List

import pandas as pd
import yaml
from p_tqdm import p_umap


def setup_args() -> argparse.Namespace:
    if Path("params.yaml").exists():
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)
    else:
        params = {
            'seed': 42,
            'create-train-test': {
                'perturbed-fraction': 0.1
            }
        }
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='Directory containing the windows')
    parser.add_argument('output', help="Output directory")
    parser.add_argument('--seed', '-s', default=params['seed'], type=int)
    parser.add_argument('--fraction', '-f',
                        default=params['create-train-test']['perturbed-fraction'],
                        type=float,
                        help="Fraction of data to be perturbed")
    return parser.parse_args()


SAFE_COLS = {"ClusterLatitude", "ClusterLongitude", "dateTimeGroup", "TimeWindowID", "Cluster"}
Edit = namedtuple("Edit", ["row", "column", "old_value", "new_value"])


def perturb(split: Path, fraction: float = 0.1, seed: int = 42) -> List[Edit]:
    history = []
    shutil.copy(split / 'test.csv', split / 'test.csv.orig')
    df = pd.read_csv(split / 'test.csv')
    unsafe_cols = list(set(df.columns) - SAFE_COLS)
    min_max = df.aggregate(['min', 'max'])
    perturbed_data = df.sample(frac=fraction, random_state=seed)
    for i, data in perturbed_data.iterrows():
        chosen_cols = random.choices(unsafe_cols, k=random.randint(1, len(unsafe_cols)))
        for col in chosen_cols:
            if str(df.dtypes[col]).startswith("int"):
                randfunc = random.randint
            elif str(df.dtypes[col]).startswith("float"):
                randfunc = random.uniform
            else:
                raise RuntimeError(f"I can't generate random values for {str(df.dtypes[col])}")
            old = data[col]
            data[col] = randfunc(min_max.loc['min', col], min_max.loc['max', col])
            history.append(Edit(row=i, column=col, old_value=old, new_value=data[col]))
        perturbed_data.loc[i] = data
    df["perturbed"] = False
    df.loc[perturbed_data.index, "perturbed"] = True
    df.update(perturbed_data)
    df.to_csv(split / 'test.csv', index=False)
    return list(filter(lambda x: x.old_value != x.new_value, history))


def generate_train_set(folder: Path, files: List[Path]):
    with open(folder / 'train.csv', 'w') as dest:
        for j, file in enumerate(files):
            with open(file, 'r') as source:
                # Skip the header if it is not the first file
                if j != 0:
                    next(source)
                # Copy the entire file in the final one
                for line in source:
                    dest.write(line)


def generate_split(output: Path, files: List[Path], i: int, fraction: float = 0.1, seed: int = 42):
    folder = output / f"split-{i}"
    folder.mkdir(exist_ok=True, parents=True)
    generate_train_set(folder, files[0:i])
    shutil.copy(files[i], folder / 'test.csv')
    with open(folder / 'contained-windows.yaml', "w") as f:
        yaml.safe_dump({
            'training-windows': [str(x) for x in files[0:i]],
            'testing-window': str(files[i])
        }, f)
    history = perturb(folder, fraction=fraction, seed=seed)
    pd.DataFrame(data=history).to_csv(folder / 'perturbations.csv', index=False)


def main():
    args = setup_args()
    directory = Path(args.directory)
    output = Path(args.output)
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(exist_ok=True, parents=True)
    files = sorted(directory.glob('window-*.csv'), key=lambda x: int(str(x.name)[7:-4]))
    func = partial(generate_split, output, files, fraction=args.fraction, seed=args.seed)
    p_umap(func, range(1, len(files)))


if __name__ == '__main__':
    main()
