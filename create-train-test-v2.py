#!/usr/bin/env python3

#  Copyright (C) 2022 Esposito Andrea and Montanaro Graziano
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import shutil
from collections import namedtuple
from functools import partial
from pathlib import Path
import random
from typing import List, Tuple, Set, Union

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
            },
            'resize': {
                'to-drop': ["monitoredCall/StopPointName", "monitoredCall/StopPointRef"]
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
    parser.add_argument('--drop', '-d', action='append')
    args = parser.parse_args()
    if not args.drop:
        args.drop = params['resize']['to-drop']
    args.drop = tuple(args.drop)
    return args


SAFE_COLS = {"id", "ClusterLatitude", "ClusterLongitude", "dateTimeGroup", "TimeWindowID", "Cluster"}
Edit = namedtuple("Edit", ["row", "column", "old_value", "new_value"])


def perturb(split: Path) -> List[Edit]:
    history = []
    shutil.copy(split / 'test.csv', split / 'test.csv.orig')
    df = pd.read_csv(split / 'test.csv', index_col="id")
    unsafe_cols = list(set(df.columns) - SAFE_COLS)
    min_max = df.aggregate(['min', 'max'])
    for i, data in df.iterrows():
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
        df.loc[i] = data
    df["perturbed"] = 1
    df.to_csv(split / 'test.csv', index=True)
    return list(filter(lambda x: x.old_value != x.new_value, history))


def generate_train_set(outfile: Path, files: Union[List[Path], Path], drop: Tuple[str] = ()):
    header = True
    files = [files] if type(files) is not list else files
    for file in files:
        reader = pd.read_csv(file, usecols=lambda x: not x.startswith(drop), chunksize=1000)
        for chunk in reader:
            chunk["perturbed"] = 0
            chunk.to_csv(outfile, index=True, index_label='id', header=header, mode='a')
        header = False


def generate_split(output: Path,
                   files: List[Path],
                   i: int,
                   to_perturb: Set[int] = None,
                   drop: Tuple[str] = ()):
    folder = output / f"split-{i}"
    folder.mkdir(exist_ok=True, parents=True)
    generate_train_set(folder / 'train.csv', files[0:i], drop=drop)
    generate_train_set(folder / 'test.csv', files[i], drop=drop)
    with open(folder / 'contained-windows.yaml', "w") as f:
        yaml.safe_dump({
            'training-windows': [str(x.name) for x in files[0:i]],
            'testing': {
                'window': str(files[i].name),
                'is-perturbed': i in to_perturb
            }
        }, f)
    if i in to_perturb:
        history = perturb(folder)
        pd.DataFrame(data=history).to_csv(folder / 'perturbations.csv', index=False)
    else:
        df = pd.read_csv(folder / 'test.csv', index_col="id")
        df["perturbed"] = 0
        df.to_csv(folder / 'test.csv', index=True)


def main():
    args = setup_args()
    random.seed(args.seed, version=2)
    directory = Path(args.directory).absolute()
    output = Path(args.output).absolute()
    shutil.rmtree(output, ignore_errors=True)
    output.mkdir(exist_ok=True, parents=True)
    files = sorted(directory.glob('window-*.csv'), key=lambda x: int(str(x.name)[7:-4]))
    to_perturb = set(random.sample(range(1, len(files)), int((len(files) - 1) * args.fraction)))
    with open(output / "perturbed-splits.yaml", 'w') as f:
        yaml.safe_dump({
            'perturbed-splits': sorted(list(to_perturb))
        }, f)
    func = partial(generate_split, output, files, to_perturb=to_perturb, drop=args.drop)
    p_umap(func, range(1, len(files)))


if __name__ == '__main__':
    main()
