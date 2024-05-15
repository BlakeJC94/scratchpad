from pytest import fixture


@fixture
def date1():
    return "2024-03-06T10:46:05"


@fixture
def date2():
    return "2024-05-06T10:46:05"


@fixture
def date3():
    return "2024-05-13T10:46:05"


@fixture
def query_fields(date2, date3):
    return {
        "format": "csv",
        "endtime": date3,
        "starttime": date2,
    }

@fixture
def query_fields_large(date1, date3):
    return {
        "format": "csv",
        "endtime": date3,
        "starttime": date1,
    }


@fixture
def query_fields_rectangle(query_fields):
    return {
        **query_fields,
        "maxlatitude": 45,
        "minlatitude": -45,
        "maxlongitude": 90,
        "minlongitude": -90,
    }


@fixture
def query_fields_circle(query_fields):
    return {
        **query_fields,
        "latitude": -45,
        "longitude": 90,
        "maxradiuskm": 58,
    }
