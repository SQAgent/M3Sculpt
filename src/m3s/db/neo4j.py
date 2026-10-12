"""Neo4j driver wrapper used by the paper retrieval agent."""

from neo4j import GraphDatabase


class Neo4jClientManager:
    """Hold one Neo4j driver and run parameterized Cypher queries."""

    def __init__(self, neo4jconfig):
        self.config = neo4jconfig
        self._get_client()

    def _get_client(self):
        """Create the Neo4j driver; leave `client` unset when it fails."""
        try:
            self.client = GraphDatabase.driver(
                self.config.url,
                auth=(self.config.username, self.config.password),
            )
            print("Neo4j client initialized")
        except Exception as e:
            self.client = None
            print(f"Neo4j client initialization failed: {e}")

    def neo4j_parameterized_query(self, cypher: str, parameters: dict) -> dict:
        """Run a parameterized Cypher query and return `{"node": [...]}`."""
        if not self.client:
            print("Neo4j client is not initialized; query cannot run")
            return {"node": []}
        try:
            with self.client.session() as session:
                print(f"Cypher query: {cypher}, parameters: {parameters}")
                return {"node": session.run(cypher, **parameters).data()}
        except Exception as e:
            print(f"Neo4j query failed: {e}")
            return {"node": []}
