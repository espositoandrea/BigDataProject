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
    # "LineRef",
    "OriginName",
    "DestinationName",
    "VehicleRef",
    "monitoredCall/DestinationDisplay",
]


def encode_dataset(df: pd.DataFrame):
    df2 = df.join(pd.get_dummies(df["OriginRef"], prefix="OriginRef", sparse=True))
    df2 = df2.join(pd.get_dummies(df2["DestinationRef"], prefix="DestinationRef", sparse=True))
    df2 = df2.join(
        pd.get_dummies(df2["monitoredCall/VehicleAtStop"], prefix="monitoredCall/VehicleAtStop", sparse=True))
    df2 = df2.join(
        pd.get_dummies(df2["monitoredCall/StopPointName"], prefix="monitoredCall/StopPointName", sparse=True))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/VisitNumber"], prefix="monitoredCall/VisitNumber", sparse=True))
    df2 = df2.join(pd.get_dummies(df2["monitoredCall/StopPointRef"], prefix="monitoredCall/StopPointRef", sparse=True))
    df2["Delay"] = df2["Delay"].map(lambda x: int(re.sub(r"(-?)PT(\d+)S", r"\1\2", x)))
    df2["LineDirectionRef"] = df2["LineRef"] + ":" + df["DirectionRef"].astype("string")
    df2.drop(columns=["LineRef", "DirectionRef"], inplace=True)
    df2 = df2.join(pd.get_dummies(df2["LineDirectionRef"], prefix="LineDirectionRef", sparse=True))
    # TODO: Check if one hot encoding is the right thing to do. It is the same as counting the "1s"
    df2 = df2.join(pd.get_dummies(df2["HeadwayService"], prefix="HeadwayService", sparse=True))
    df2["DestinationAimedArrivalTime"] = df2["DestinationAimedArrivalTime"].map(lambda x: x.tz_localize(None))
    df2["OriginAimedDepartureTime"] = df2["OriginAimedDepartureTime"].map(lambda x: x.tz_localize(None))
    return df2.sort_values('dateTime')


def get_vectorial_aggregations(columns: List[str]) -> Dict[str, str]:
    for col in columns:
        if col.startswith(('DestinationRef_', 'OriginRef_', 'monitoredCall/VehicleAtStop_',
                           'monitoredCall/StopPointName_', 'monitoredCall/VisitNumber_',
                           'monitoredCall/StopPointRef_', 'LineDirectionRef_', 'HeadwayService_')):
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
                     parse_dates=["dateTime", "OriginAimedDepartureTime", "DestinationAimedArrivalTime"])

    # This has some Null values: how can we treat it? We should make this code more general
    df.drop(labels=df[df["Delay"].isnull()].index, inplace=True)

    df = encode_dataset(df)

    df["dateTimeDiff"] = df["dateTime"].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds().astype('int')

    indexer = df["dateTime"].sort_values().diff().fillna(pd.Timedelta(seconds=0)).cumsum()
    grouper = indexer.map(lambda x: x.floor('5T').total_seconds() / 60).astype('int').rename("TimeWindowID")
    df["dateTimeGroup"] = df["dateTime"].map(lambda x: x.floor(freq='5T'))

    def to_seconds(x: pd.Series) -> pd.Series:
        return (x - pd.Timestamp(0)).dt.total_seconds().astype('int')

    df["OriginAimedDepartureTime"] = to_seconds(df["OriginAimedDepartureTime"])
    df["DestinationAimedArrivalTime"] = to_seconds(df["DestinationAimedArrivalTime"])

    converted_times = pd.concat(
        group[["OriginAimedDepartureTime", "DestinationAimedArrivalTime"]].diff().fillna(0) for _, group in
        df.groupby(grouper))
    df[["OriginAimedDepartureTime", "DestinationAimedArrivalTime"]] = converted_times

    grp = df.groupby([grouper, 'Cluster'])
    agg = grp.aggregate({
        'Cluster Latitude': 'min',
        'Cluster Longitude': 'min',
        'dateTimeGroup': 'min',
        'dateTimeDiff': 'mean',
        'Delay': 'mean',
        'Percentage': 'mean',
        'InPanic': 'sum',
        'InCongestion': 'sum',
        'DestinationAimedArrivalTime': 'mean',
        'OriginAimedDepartureTime': 'mean',
        **dict(get_vectorial_aggregations(list(df.columns)))
    })
    agg.to_csv(sys.stdout)
    pass


if __name__ == '__main__':
    main()
