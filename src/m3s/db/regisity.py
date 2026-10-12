from m3s.db.milvus import MilvusClientManager
from m3s.db.mongo import MongoClientManager
from m3s.db.neo4j import Neo4jClientManager

DataBase_CLIENT_MAP = {
    "neo4j": Neo4jClientManager,
    "mongo": MongoClientManager,
    "milvus": MilvusClientManager,
}