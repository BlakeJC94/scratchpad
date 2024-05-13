from datetime import datetime


def check_time_field_is_valid(time_value: str):
    try:
        _ = datetime.fromisoformat(time_value)
    except Exception as exc:
        raise ValueError(f"Invalid time given (received {time_value})") from exc
