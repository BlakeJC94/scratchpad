from datetime import timedelta
BASE_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

RESPONSE_BAD_REQUEST = 400
RESPONSE_NO_CONTENT = 204
RESPONSE_NOT_FOUND = 404
RESPONSE_OK = 200

SEGMENT_SECS = timedelta(days=24).total_seconds()

NUM_WORKERS = 6
MAX_ATTEMPTS = 5
