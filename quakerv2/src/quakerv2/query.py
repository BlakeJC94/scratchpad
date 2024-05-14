from copy import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from math import ceil
from typing import Any

from quakerv2.globals import SEGMENT_SECS
from quakerv2.utils import check_time_field_is_valid


@dataclass
class Query:
    endtime: str = field(default=None)
    starttime: str = field(default=None)
    updatedafter: str = field(default=None)
    maxdepth: float = field(default=None)
    mindepth: float = field(default=None)
    maxmagnitude: float = field(default=None)
    minmagnitude: float = field(default=None)

    def __post_init__(self):
        self.orderby = "time"
        self.format = "csv"
        self.validate()

    def validate(self):
        for time_field in ["starttime", "endtime", "updatedafter"]:
            if (time := getattr(self, time_field)) is not None:
                check_time_field_is_valid(time)

    def copy(self):
        return copy(self)

    def dict(self, include_nones: bool = False) -> dict[str, Any]:
        query_dict = vars(self)
        if include_nones:
            return query_dict
        return {k: v for k, v in query_dict.items() if v is not None}


@dataclass
class QueryRectangle(Query):
    minlatitude: float = field(default=None)
    maxlatitude: float = field(default=None)
    minlongitude: float = field(default=None)
    maxlongitude: float = field(default=None)


@dataclass
class QueryCircle(Query):
    latitude: float = field(default=None)
    longitude: float = field(default=None)
    maxradius: float = field(default=None)
    maxradiuskm: float = field(default=None)


def get_query(**kwargs: dict[str, Any]) -> Query:
    rectangle_kwargs = [
        "minlatitude",
        "maxlatitude",
        "minlongitude",
        "maxlongitude",
    ]
    circle_kwargs = [
        "latitude",
        "longitude",
        "maxradius",
        "maxradiuskm",
    ]

    if any(k in kwargs for k in rectangle_kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in circle_kwargs}
        return QueryRectangle(**kwargs)

    if any(k in kwargs for k in circle_kwargs):
        kwargs = {k: v for k, v in kwargs.items() if k not in rectangle_kwargs}
        return QueryCircle(**kwargs)

    return Query(**kwargs)


def split_query(query: Query) -> list[Query]:
    orderby = query.orderby
    starttime = query.starttime
    starttime = (
        datetime.fromisoformat(starttime)
        if starttime is not None
        else datetime.now() - timedelta(days=30)
    )

    endtime = query.endtime
    endtime = datetime.fromisoformat(endtime) if starttime is not None else datetime.now()

    n_segments = ceil((endtime - starttime).total_seconds() / SEGMENT_SECS)

    sub_queries = []
    for i in range(n_segments):
        sub_query = query.copy()
        sub_query.starttime = max(
            starttime, starttime + i * timedelta(seconds=SEGMENT_SECS)
        ).isoformat()
        sub_query.endtime = min(
            endtime, starttime + (i + 1) * timedelta(seconds=SEGMENT_SECS)
        ).isoformat()
        sub_queries.append(sub_query)

    if orderby == "time":
        sub_queries = sub_queries[::-1]

    return sub_queries
