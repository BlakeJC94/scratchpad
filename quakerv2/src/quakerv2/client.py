import random
import logging
from time import sleep

from requests.sessions import Session

from quakerv2.globals import (
    BASE_URL,
    MAX_ATTEMPTS,
    NUM_WORKERS,
    RESPONSE_BAD_REQUEST,
    RESPONSE_NOT_FOUND,
)
from quakerv2.query import Query, get_query
from quakerv2.file import get_file, join_files


class Client:
    def __init__(self):
        self.num_workers = NUM_WORKERS

        self.session = Session()
        self.logger = logging.getLogger(__name__)

    def execute(self, **kwargs):
        query = get_query(**kwargs)
        return self._execute_sq(query)

    def _execute_sq(self, query: Query) -> str:
        query.limit = 20000

        pages = []
        fetch_next_page = True
        while fetch_next_page:
            page = get_file(query.format, self._execute(query))
            pages.append(page)
            if len(page.records()) <= 20000:
                fetch_next_page = False
            else:
                query.offset += 20000

        return join_files(pages).content

    def _execute(self, query: Query) -> str:
        with self.session as session:
            for idx in range(MAX_ATTEMPTS):
                sleep(random.expovariate(1 + idx * 0.5))
                response = session.get(BASE_URL, params=query.dict())

                if response.status_code != RESPONSE_NOT_FOUND:
                    self._check_download_error(response)
                    return response.text.strip()

                self.logger.warning(f"No connection could be made, retrying ({idx}).")

        raise ConnectionAbortedError("Connection could not be established")

    def _check_download_error(self, response):
        if response.ok:
            return

        status = response.status_code
        msg = f"Unexpected response code on query ({status})."
        if status == RESPONSE_BAD_REQUEST:
            msg = f"Invalid query ({RESPONSE_BAD_REQUEST})."

        self.logger.error(msg)
        raise RuntimeError(msg)
