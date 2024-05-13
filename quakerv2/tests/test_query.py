from math import ceil
from datetime import datetime

from quakerv2.globals import SEGMENT_SECS
from quakerv2.query import Query, QueryCircle, QueryRectangle, get_query, split_query


def test_query(query_fields, date1):
    query = get_query(**query_fields)

    assert isinstance(query, Query)
    assert query.starttime == query_fields["starttime"]
    assert query.endtime == query_fields["endtime"]
    assert query.updatedafter is None

    result = query.dict()
    assert result == {**query_fields, "format": "csv", "orderby": "time"}

    query_copy = query.copy()
    query_copy.starttime = date1
    query_copy.updatedafter = date1

    assert query_copy.starttime == date1
    assert query_copy.endtime == query_fields["endtime"]
    assert query_copy.updatedafter == date1


def test_query_rectangle(query_fields_rectangle):
    query = get_query(**query_fields_rectangle)

    assert isinstance(query, Query)
    assert isinstance(query, QueryRectangle)

    result = query.dict()
    assert result == {**query_fields_rectangle, "format": "csv", "orderby": "time"}


def test_query_circle(query_fields_circle):
    query = get_query(**query_fields_circle)

    assert isinstance(query, Query)
    assert isinstance(query, QueryCircle)

    result = query.dict()
    assert result == {**query_fields_circle, "format": "csv", "orderby": "time"}


def test_split_query(query_fields_large):
    query = Query(**query_fields_large)
    sub_queries = split_query(query)

    assert all(isinstance(q, Query) for q in sub_queries)
    assert sub_queries[-1].starttime == query.starttime
    assert sub_queries[0].endtime == query.endtime
    assert len(sub_queries) == ceil(
        (datetime.fromisoformat(query.endtime) - datetime.fromisoformat(query.starttime)).total_seconds() / SEGMENT_SECS
    )
