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
import functools
import sys
from typing import Tuple, List

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import yaml


def confusion_matrix(group: Tuple[pd.Timestamp, pd.DataFrame]) -> Tuple[int, int, int, int]:
    _, grp = group
    tp = grp.loc[(grp['Prediction'] == 1) & (grp['Actual'] == 1)].count()['Tau']
    tn = grp.loc[(grp['Prediction'] == 0) & (grp['Actual'] == 0)].count()['Tau']
    fp = grp.loc[(grp['Prediction'] == 1) & (grp['Actual'] == 0)].count()['Tau']
    fn = grp.loc[(grp['Prediction'] == 0) & (grp['Actual'] == 1)].count()['Tau']
    return tp, tn, fp, fn


def split_to_datetime_label(split: int, perturbed: List[int] = None) -> str:
    # indicator = '' if perturbed and split in perturbed else ''
    # format_str = "%Y-%m-%d %H:%M"
    format_str = "%m-%d %H:%M"
    return ((pd.Timestamp("2022-03-01T18:40:00") + pd.Timedelta(hours=split - 1))
            .strftime(format_str)
            .strip())


def get_test_data(index: pd.Index) -> pd.DataFrame:
    import numpy
    return pd.DataFrame(index=index,
                        columns=['tp', 'tn', 'fp', 'fn'],
                        data=numpy.random.randint(1, high=15, size=(len(index), 4)))


def generate_missing_splits(indf):
    limit = max(indf)
    sizes = [
            (1, 295), (2, 279), (3, 268), (4, 265),
            (5, 327), (6, 251), (7, 188), (8, 41),
            (9, 7), (10, 72), (11, 154), (12, 292),
            (13, 363), (14, 365), (15, 377), (16, 320),
            (17, 322), (18, 360), (19, 379), (20, 416),
            (21, 403), (22, 415), (23, 448), (24, 444),
            (25, 366), (26, 346), (27, 357), (28, 355),
            (29, 332), (30, 308), (31, 211), (32, 45),
            (33, 28), (34, 75), (35, 167), (36, 331),
            (37, 457), (38, 490), (39, 491), (40, 396),
            (41, 386), (42, 389), (43, 396), (44, 398),
            (45, 444), (46, 429), (47, 435), (48, 443),
            (49, 360), (50, 342), (51, 326), (52, 327),
            (53, 315), (54, 300), (55, 220), (56, 46),
            (57, 31), (58, 79), (59, 170), (60, 336),
            (61, 439), (62, 471), (63, 454), (64, 408),
            (65, 393), (66, 395), (67, 392), (68, 435),
            (69, 440), (70, 460), (71, 443), (72, 429),
            (73, 367), (74, 339), (75, 320), (76, 307),
            (77, 287), (78, 290), (79, 232), (80, 161),
            (81, 146), (82, 122), (83, 80), (84, 129),
            (85, 196), (86, 224), (87, 259), (88, 320),
            (89, 335), (90, 333), (91, 333), (92, 324),
            (93, 342), (94, 343), (95, 365), (96, 378),
            (97, 365), (98, 347), (99, 329), (100, 295),
            (101, 292), (102, 277), (103, 218), (104, 156),
            (105, 125), (106, 132), (107, 81), (108, 116),
            (109, 170), (110, 198), (111, 218), (112, 243),
            (113, 242), (114, 272), (115, 272), (116, 277),
            (117, 273), (118, 282), (119, 277), (120, 277),
            (121, 280), (122, 309), (123, 325), (124, 277),
            (125, 288), (126, 266), (127, 195), (128, 44),
            (129, 25), (130, 73), (131, 165), (132, 327),
            (133, 429), (134, 467), (135, 473), (136, 412),
            (137, 394), (138, 377), (139, 402), (140, 429),
            (141, 454), (142, 459), (143, 444), (144, 455),
            (145, 353), (146, 337), (147, 324), (148, 313),
            (149, 306), (150, 275), (151, 204), (152, 44),
            (153, 30), (154, 75), (155, 176), (156, 360),
            (157, 506), (158, 512), (159, 483), (160, 437),
            (161, 418), (162, 412), (163, 414), (164, 461),
            (165, 501), (166, 506), (167, 497), (168, 410),
    ]
    sizes = set(filter(lambda x: x[0] < limit and x[0] not in indf, sizes))
    segments = [[[0.7, 100, split, 0, 0]] * size for split, size in sizes]
    return [e for l in segments for e in l]
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--perturbed', '-p', type=argparse.FileType('r'), default=None)
    args = parser.parse_args()

    perturbed_splits = yaml.safe_load(args.perturbed)['perturbed-splits'] if args.perturbed else []

    df = pd.read_csv(args.input)
    df = pd.concat([df, pd.DataFrame(generate_missing_splits(df['Split'].unique()), columns=df.columns)])
    df = df.sort_values('Split')
    df = df.loc[df['Tau'] == 0.7]

    split_to_datetime_label = functools.partial(split_to_datetime_label, perturbed=perturbed_splits)
    df['datetime'] = df['Split'].map(split_to_datetime_label)

    new_df = pd.DataFrame(index=df['datetime'].unique(),
                          columns=['tp', 'tn', 'fp', 'fn'],
                          data=map(confusion_matrix, df.groupby('datetime')))

    # new_df = get_test_data(new_df.index)

    colors = ['#2C712D', '#67C468', '#C00000', '#D1480B']
    fig, ax = plt.subplots()

    new_df['total'] = new_df['tn'] + new_df['fn'] + new_df['tp'] + new_df['fp']
    cumulative = new_df[['tn', 'fn']].add(
        new_df[['tp', 'fp']].rename(columns={'tp': 'tn', 'fp': 'fn'})
    ).divide(new_df['total'], axis='index')
    errors = new_df[['tn', 'fn']].divide(new_df['total'], axis='index')

    cumulative.plot(kind='bar', ax=ax, color=[colors[0], colors[2]])
    errors.plot(kind='bar', ax=ax, color=[colors[1], colors[3]])

    for i, label in enumerate(new_df.index):
        ax.annotate(
            new_df.loc[label, "total"],
            (i, 1.05),
            ha="center",
            va="bottom",
        )

    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    perturbed_labels = set(map(split_to_datetime_label, perturbed_splits))
    list(map(lambda t: t.set_color('#C00000'),
             filter(lambda t: t.get_text() in perturbed_labels, ax.get_xticklabels())))
    ax.legend(['TP', 'FP', 'TN', 'FN'], loc="lower center", bbox_to_anchor=(0.5, 1.15), ncol=4)

    plt.tight_layout()
    plt.show()
