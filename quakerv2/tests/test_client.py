import io
from typing import Any

import pandas as pd
from dateutil.parser import isoparse

from quakerv2.client import Client


def check_valid_csv(result: str, query_fields: dict[str, Any]):
    result = pd.read_csv(io.StringIO(result))

    # Check that all columns are present
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

    # Check non-empty result
    assert len(result) > 0

    # Check non-empty result
    assert result["time"].dtype == "object"
    assert result["mag"].dtype == "float64"

    # Check all times lie within requested time span
    dt_col = pd.to_datetime(result["time"])
    assert (dt_col.dt.to_pydatetime() <= isoparse(query_fields["endtime"] + "Z")).all()
    assert (isoparse(query_fields["starttime"] + "Z") <= dt_col.dt.to_pydatetime()).all()

    orderby = query_fields.get("orderby", "time")
    if orderby == "time":
        assert (dt_col.sort_values(ascending=False) == dt_col).all()
    elif orderby == "time-asc":
        assert (dt_col.sort_values(ascending=True) == dt_col).all()
    elif orderby == "magnitude":
        assert (result["mag"].sort_values(ascending=False) == result["mag"]).all()
    elif orderby == "magnitude":
        assert (result["mag"].sort_values(ascending=True) == result["mag"]).all()
    else:
        raise AssertionError()


def test_client(query_fields):
    client = Client()
    out = client.execute(**query_fields)
    assert len(out.split("\n")) <= 20000
    check_valid_csv(out, query_fields)


def test_client_paginated(query_fields_large):
    client = Client()
    out = client.execute(**query_fields_large)
    assert len(out.split("\n")) > 20000
    check_valid_csv(out, query_fields_large)
