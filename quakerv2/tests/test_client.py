import io
from datetime import datetime
from typing import Any

import pandas as pd

from quakerv2.client import Client


def check_valid_csv(result: str, query_fields: dict[str, Any]):
    result = pd.read_csv(io.StringIO(result))

    assert set(result.columns) == {
        "time",
        "latitude",
        "longitude",
        "depth",
        "mag",
        "magType",
        "nst",
        "gap",
        "dmin",
        "rms",
        "net",
        "id",
        "updated",
        "place",
        "type",
        "horizontalError",
        "depthError",
        "magError",
        "magNst",
        "status",
        "locationSource",
        "magSource",
    }

    assert len(result) > 0

    dt_col = pd.to_datetime(result["time"])
    assert (dt_col.sort_values(ascending=False) == dt_col).all()

    assert datetime.fromisoformat(
        result["time"].iloc[0].removesuffix("Z")
    ) <= datetime.fromisoformat(query_fields["endtime"])
    assert datetime.fromisoformat(
        result["time"].iloc[-1].removesuffix("Z")
    ) >= datetime.fromisoformat(query_fields["starttime"])


def test_client(query_fields):
    client = Client()
    out = client.execute(**query_fields)
    assert len(out.split("\n")) <= 20000
    check_valid_csv(out, query_fields)


def test_client_mt(query_fields_large):
    client = Client()
    out = client.execute(**query_fields_large)
    assert len(out.split("\n")) > 20000
    check_valid_csv(out, query_fields_large)
