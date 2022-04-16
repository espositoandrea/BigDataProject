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
import sys
from pathlib import Path
import pandas as pd
import re
from p_tqdm import p_umap


def process_result(file: Path) -> pd.DataFrame:
    match = re.match(r".*?tau1_(.*?)_epochs_(\d+?)datasets/.*?/split-(\d+?)/.*",
                     str(file))
    if not match:
        raise AssertionError()

    tau1 = float(match.group(1))
    epochs = int(match.group(2))
    split = int(match.group(3))

    cols = ["prediction", "actual"]
    df = pd.read_csv(file, header=None, names=cols)
    if (df.iloc[0] == cols).all():
        df = df[1:].reset_index(drop=True)
    df = df.apply(pd.to_numeric, downcast='integer')
    df = df.rename(columns={k: k.title() for k in cols})
    df.insert(0, 'Tau', tau1)
    df.insert(1, 'Epochs', epochs)
    df.insert(2, 'Split', split)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
            description="This script can be used to merge all the results and"
            " prediction provided by SparkGHSOM in a single CSV for further usage"
            " (e.g, in Excel or in other scripts). No modifications are applied"
            " to the initial data, but some columns are added to the final CSV to"
            " represent some information that would be encoded by the output"
            " path. Note that this was designed to be used only with the Oslo"
            " dataset."
    )
    parser.add_argument('infolder',
                        help="The folder containing SparkGHSOM output")
    parser.add_argument('--out', '-o',
                        type=argparse.FileType('w'),
                        help="The name of the output file. Defaults to stdout",
                        default=sys.stdout)
    args = parser.parse_args()

    files = list(Path(args.infolder).glob("tau*datasets/oslo/split-*/test.csv.predictions"))
    pd.concat(p_umap(process_result, files)).to_csv(args.out, index=False)
