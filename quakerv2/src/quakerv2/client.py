import concurrent.futures as cf
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
from quakerv2.query import Query, get_query, split_query


class Client:
    def __init__(self):
        self.num_workers = NUM_WORKERS

        self.session = Session()
        self.logger = logging.getLogger(__name__)

    def execute(self, **kwargs):
        query = get_query(**kwargs)
        try:
            result = self._execute(query)
        except RuntimeError:
            result = self._execute_mt(query)
        return result

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

    def _execute_mt(self, query: Query) -> str:
        sub_queries = split_query(query)

        results = ["" for _ in range(len(sub_queries))]
        with cf.ThreadPoolExecutor(max_workers=self.num_workers) as pool:
            future_to_idx = {
                pool.submit(self._execute, sub_query): i for i, sub_query in enumerate(sub_queries)
            }

            for future in cf.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    data = future.result()
                except Exception as exc:
                    self.logger.error(f"{idx} generated an exception: {exc}")
                    continue

                if idx != 0:
                    data = "\n".join(data.split("\n")[1:])

                results[idx] = data

        return "\n".join(results)

    def _check_download_error(self, response):
        if response.ok:
            return

        status = response.status_code
        msg = f"Unexpected response code on query ({status})."
        if status == RESPONSE_BAD_REQUEST:
            msg = f"Invalid query ({RESPONSE_BAD_REQUEST})."

        self.logger.error(msg)
        raise RuntimeError(msg)
