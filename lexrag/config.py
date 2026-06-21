from typing import Any
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # QDRANT SETTINGS
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "lexrag_docs"
    qdrant_vector_size: int = 1024

    # NEO4J SETTINGS
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "lexragpassword"

    # REDIS SETTINGS
    redis_url: str = "redis://localhost:6379/0"

    # GROQ SETTINGS
    groq_api_key: str = ""
    groq_model_name: str = "llama3-70b-8192"
    groq_temperature: float = 0.0
    groq_max_tokens: int = 2048

    # OLLAMA SETTINGS
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # LANGSMITH SETTINGS
    langchain_tracing_v2: str = "false"
    langchain_api_key: str = ""
    langchain_project: str = "lexrag"

    # ARIZE PHOENIX SETTINGS
    arize_endpoint: str = "http://localhost:6006/v1/traces"

    # APPLICATION SETTINGS
    app_environment: str = "development"
    app_log_level: str = "INFO"
    app_max_file_size_mb: int = 50
    app_supported_extensions: str = ".pdf,.docx,.pptx,.xlsx,.html,.md"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
