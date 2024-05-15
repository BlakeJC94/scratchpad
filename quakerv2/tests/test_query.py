from quakerv2.query import Query


def test_query(query_fields, date1):
    query = Query(**query_fields)

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
    query = Query(**query_fields_rectangle)

    assert isinstance(query, Query)

    result = query.dict()
    assert result == {**query_fields_rectangle, "format": "csv", "orderby": "time"}


def test_query_circle(query_fields_circle):
    query = Query(**query_fields_circle)

    assert isinstance(query, Query)

    result = query.dict()
    assert result == {**query_fields_circle, "format": "csv", "orderby": "time"}
