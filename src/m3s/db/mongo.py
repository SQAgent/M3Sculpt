"""MongoDB client wrapper for the structure-property collection."""

from pymongo import MongoClient


class MongoClientManager:
    """Hold one MongoDB database handle for property lookups."""

    def __init__(self, mongoconfig):
        self.config = mongoconfig
        self._create_client()

    def _create_client(self):
        """Connect to MongoDB and keep the configured database handle."""
        self.client = MongoClient(self.config.url)[self.config.database]
        print("MongoDB client initialized")
