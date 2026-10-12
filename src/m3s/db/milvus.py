"""Milvus client wrapper used for abstract-level semantic retrieval."""

import math

from pymilvus import MilvusClient


class MilvusClientManager:
    """Hold one Milvus client and run vector searches over paper abstracts."""

    ALLOWED_KEYS = ("collection_name", "data", "top_k", "search_params", "output_fields")

    def __init__(self, milvusconfig):
        self.config = milvusconfig
        self.query_kword = {
            "collection_name": "entity",
            "data": None,
            "top_k": 5,
            "search_params": {"metric_type": "COSINE", "params": {}},
            "output_fields": ["id"],
        }
        self._create_client()

    def _create_client(self):
        """Connect to Milvus and select the configured database."""
        token = f"{self.config.username}:{self.config.password}"
        self.client = MilvusClient(uri=self.config.url, token=token)
        print(f"Milvus client initialized; connected to {self.config.url}")
        self.client.use_database(db_name=self.config.database)

    def update(self, key: str, value) -> None:
        """Update one allowed search parameter; other keys are rejected."""
        if key in self.ALLOWED_KEYS:
            self.query_kword[key] = value
        else:
            print(f"MilvusConfig does not allow updating key '{key}'.")

    def milvus_query(self) -> list:
        """Run the configured vector search and return the raw result list."""
        results = self.client.search(
            collection_name=self.query_kword["collection_name"],
            data=self.query_kword["data"],
            limit=self.query_kword["top_k"],
            search_params=self.query_kword["search_params"],
            output_fields=self.query_kword["output_fields"],
        )
        print(f"Number of Milvus query results: {len(results)}")
        return results

    def semantic_score(self, distance) -> float:
        """Convert a raw hit to a score where larger means more similar.

        Milvus calls the returned field ``distance`` for all metrics. COSINE
        and IP are similarities; L2 is a squared distance. Its reciprocal
        mapping keeps the score nonnegative and preserves nearest-first order.
        """
        metric = str(self.query_kword["search_params"].get("metric_type", "")).upper()
        if metric not in {"COSINE", "IP", "L2"}:
            raise ValueError(f"Unsupported Milvus semantic metric: {metric!r}")
        try:
            value = float(distance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid Milvus {metric} score: {distance!r}") from exc
        if isinstance(distance, bool) or not math.isfinite(value):
            raise ValueError(f"Invalid Milvus {metric} score: {distance!r}")
        if metric == "L2":
            if value < 0:
                raise ValueError(f"Milvus L2 distance must be nonnegative: {value}")
            return 1.0 / (1.0 + value)
        return value
