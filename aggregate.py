#!/usr/bin/env python3
import argparse
import re
import sys
from typing import Dict, List

import pandas as pd


def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    return parser.parse_args()


DROP_COLS = [
    "LinkDistance",
    "LineRef",
    "OriginName",
    "DestinationName",
    "VehicleRef",
    "monitoredCall/DestinationDisplay",
]


def encode_dataset(df: pd.DataFrame):
    df2 = df.join(pd.get_dummies(df["OriginRef"], prefix="OriginRef"))
    df2 = df2.join(pd.get_dummies(df2["DestinationRef"], prefix="DestinationRef"))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/VehicleAtStop"], prefix="monitoredCall/VehicleAtStop"))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/StopPointName"], prefix="monitoredCall/StopPointName"))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/VisitNumber"], prefix="monitoredCall/VisitNumber"))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/StopPointRef"], prefix="monitoredCall/StopPointRef"))
    df2["Delay"] = df2["Delay"].map(lambda x: int(re.sub(r"(-?)PT(\d+)S", r"\1\2", x)))
    return df2


def get_vectorial_aggregations(columns: List[str]) -> Dict[str, str]:
    for col in columns:
        if col.startswith(('DestinationRef_', 'OriginRef_', 'monitoredCall/VehicleAtStop_',
                           'monitoredCall/StopPointName_', 'monitoredCall/VisitNumber_',
                           'monitoredCall/StopPointRef_')):
            yield col, 'sum'


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

    # This has some Null values: how can we treat it? We should make this code more general
    df.drop(labels=62, inplace=True)

    df = encode_dataset(df)

    indexer = df["dateTime"].sort_values().diff().fillna(pd.Timedelta(seconds=0)).cumsum()
    grouper = indexer.map(lambda x: x.floor('5T').total_seconds() / 60).astype('int').rename("TimeWindowID")
    df["dateTimeGroup"] = df["dateTime"].map(lambda x: x.floor(freq='T'))

    grp = df.groupby([grouper, 'Cluster'])
    agg = grp.aggregate({
        'Cluster Latitude': 'min',
        'Cluster Longitude': 'min',
        'dateTimeGroup': 'min',
        'Delay': 'mean',
        'Percentage': 'mean',
        'InPanic': 'sum',
        'InCongestion': 'sum',
        **dict(get_vectorial_aggregations(list(df.columns)))
    })
    agg.to_csv(sys.stdout)
    pass


if __name__ == '__main__':
    main()
