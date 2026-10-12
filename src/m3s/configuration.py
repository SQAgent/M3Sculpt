# -*- coding: utf-8 -*-
"""Configuration for the RAG retrieval databases and the 3D reconstruction client."""

import os
import random
from dataclasses import InitVar, dataclass
from typing import Any, Dict, Optional

from m3s.db.regisity import DataBase_CLIENT_MAP


@dataclass
class HunyuanConfig:
    """Configure the local Hunyuan3D API and its request parameters."""

    url: str = "http://localhost:7860/generate"
    output_path: str = os.path.join(os.getcwd(), "log", "hunyuan")  # Generated model output directory
    image_path: Optional[str] = None    # Image file path
    remove_background: bool = True      # Remove image background
    texture: bool = False               # Generate textures
    type: str = "glb"                   # Generated model format (glb)
    seed: Optional[int] = None          # Random seed
    octree_depth: int = 5               # Octree depth controlling model detail, range 1-7
    guidance_scale: float = 7.5         # Guidance scale, range 0.1-20
    face_count: int = 5000              # Maximum texture-generation face count, range 1000-100000
    image: InitVar[Optional[str]] = None  # Base64-encoded image string (initialization only)

    def __post_init__(self, image: Optional[str] = None):
        """Generate a random seed when none is specified."""
        if self.seed is None:
            self.seed = random.randint(0, 2**32 - 1)
        print(f"Using random seed: {self.seed}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert HunyuanConfig to a dictionary."""
        return {
            "url": self.url,
            "output_path": self.output_path,
            "image_path": self.image_path,
            "remove_background": self.remove_background,
            "texture": self.texture,
            "type": self.type,
            "seed": self.seed,
            "octree_depth": self.octree_depth,
            "guidance_scale": self.guidance_scale,
            "face_count": self.face_count,
        }


@dataclass
class Neo4jConfig:
    """Configure the Neo4j database."""

    url: str
    username: str
    password: str
    database: str
    batch_size_nodes: int = 500
    batch_size_edges: int = 100
    client: Any = None

    def __post_init__(self):
        """Initialize Neo4j; leave the client unset on failure for the caller to handle."""
        try:
            self.client = DataBase_CLIENT_MAP["neo4j"](self)
        except Exception:
            self.client = None


@dataclass
class MongoConfig:
    """Configure the MongoDB database."""

    url: str
    database: str
    client: Any = None

    def __post_init__(self):
        """Initialize MongoDB; leave the client unset on failure for the caller to handle."""
        try:
            self.client = DataBase_CLIENT_MAP["mongo"](self)
        except Exception:
            self.client = None


@dataclass
class MilvusConfig:
    """Configure the Milvus vector database."""

    url: str
    username: str
    password: str
    database: str
    client: Any = None

    def __post_init__(self):
        """Initialize Milvus; leave the client unset on failure for the caller to handle."""
        try:
            self.client = DataBase_CLIENT_MAP["milvus"](self)
        except Exception:
            self.client = None


@dataclass
class DatabaseConfig:
    """Hold the configuration for all retrieval databases."""

    neo4j: Neo4jConfig
    mongo: MongoConfig
    milvus: MilvusConfig

    @classmethod
    def from_dict(cls, config: dict) -> "DatabaseConfig":
        """Create a configuration instance from a dictionary."""
        return cls(
            neo4j=Neo4jConfig(**config.get("neo4j", {})),
            mongo=MongoConfig(**config.get("mongo", {})),
            milvus=MilvusConfig(**config.get("milvus", {})),
        )


def create_database_config_from_env() -> DatabaseConfig:
    """Create DatabaseConfig from environment variables."""
    return DatabaseConfig(
        neo4j=Neo4jConfig(
            url=os.environ.get("NEO4J_URL"),
            username=os.environ.get("NEO4J_USERNAME"),
            password=os.environ.get("NEO4J_PASSWORD"),
            database=os.environ.get("NEO4J_DATABASE"),
        ),
        mongo=MongoConfig(
            url=os.environ.get("MONGO_URL"),
            database=os.environ.get("MONGO_DATABASE"),
        ),
        milvus=MilvusConfig(
            url=os.environ.get("MILVUS_URL"),
            username=os.environ.get("MILVUS_USERNAME"),
            password=os.environ.get("MILVUS_PASSWORD"),
            database=os.environ.get("MILVUS_DATABASE"),
        ),
    )
