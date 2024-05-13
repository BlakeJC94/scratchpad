import pytest

from quakerv2.utils import check_time_field_is_valid


@pytest.mark.parametrize("time", ["2024-05-06T12:34:56", "2024-03-04"])
def test_check_time_field_is_valid(time):
    check_time_field_is_valid(time)


@pytest.mark.parametrize("invalid_time", ["foo", "01-01-2020"])
def test_check_time_field_is_valid_invalid_times(invalid_time):
    with pytest.raises(ValueError):
        check_time_field_is_valid(invalid_time)
