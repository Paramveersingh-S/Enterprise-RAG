<div align="center">

# LexRAG — Enterprise Legal Document Intelligence Platform

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-EF3D54)](https://qdrant.tech/)
[![Neo4j](https://img.shields.io/badge/Neo4j-Graph%20DB-018bff)](https://neo4j.com/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)](https://python.langchain.com/docs/langgraph)
[![Groq](https://img.shields.io/badge/Groq-LLM-f55036)](https://groq.com/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white)](https://www.docker.com/)

An enterprise-grade Retrieval-Augmented Generation (RAG) system built from scratch, demonstrating deep mastery of the modern AI stack with hybrid search, graph reasoning, and production observability.

</div>

## 🏗️ Architecture

```mermaid
graph TD
    %% Ingestion Pipeline
    subgraph Ingestion["Document Ingestion Pipeline"]
        Doc[Documents PDF/Word/HTML] --> Parsers[Multi-modal Parsers Unstructured / Docling]
        Parsers --> Dedup[Redis Deduplication]
        Dedup --> Chunkers[Hierarchical / Semantic Chunkers]
    end

    %% Embedding & Indexing
    subgraph Indexing["Embedding & Indexing"]
        Chunkers --> Embed[BGE-M3 Dual Encoder via Ollama]
        Embed --> Dense[Dense Vectors]
        Embed --> Sparse[Sparse Vectors BM25]
        Dense --> Qdrant[(Qdrant Vector DB)]
        Sparse --> Qdrant
        
        Chunkers --> NER[spaCy Entity Extractor]
        NER --> Neo4j[(Neo4j Graph DB)]
    end

    %% Retrieval & Orchestration
    subgraph Orchestration["Agentic Orchestration (LangGraph)"]
        Query[User Query] --> Analyze[Query Analysis Node]
        Analyze --> HSearch[Hybrid Vector Search RRF]
        Analyze --> GSearch[Graph Entity Search]
        
        HSearch --> Qdrant
        GSearch --> Neo4j
        
        Qdrant --> Rerank[Cross-Encoder Reranker]
        Neo4j --> Rerank
        
        Rerank --> Gen[LLM Generation Node Groq]
        Gen --> Validate[Validation Node]
        Validate --"Fail"--> Analyze
        Validate --"Pass"--> Output[Final Answer + Citations]
    end

    %% Observability
    subgraph Observability["Observability"]
        Orchestration -.-> Phoenix[Arize Phoenix]
        Orchestration -.-> LangSmith[LangSmith]
        Orchestration -.-> Grafana[Grafana Metrics]
    end
```

## ✨ Enterprise Features

- **Robust Ingestion**: Handles native PDFs, scanned PDFs (OCR fallback), Word, Excel, PowerPoint, and HTML. Includes Redis-based deduplication and SQLite failure logging.
- **Advanced Chunking**: Hierarchical (small-to-big) and semantic chunking using `tiktoken` for accurate token limits.
- **Hybrid Search**: Combines dense embeddings and BM25 sparse vectors using Reciprocal Rank Fusion (RRF).
- **Graph RAG**: Extracts entities and relationships via spaCy into Neo4j for multi-hop reasoning.
- **Cross-Encoder Re-ranking**: Uses BGE Reranker v2 to dramatically improve precision on the top-20 retrieved chunks.
- **Agentic Orchestration**: LangGraph-based state machine that self-corrects and iterates on ambiguous queries.
- **Production Observability**: Full OpenTelemetry tracing with Arize Phoenix, LangSmith, and Prometheus/Grafana metrics.

## 🚀 Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/Paramveersingh-S/Enterprise-RAG.git
   cd Enterprise-RAG
   ```

2. **Set up environment**
   ```bash
   cp .env.example .env
   # Edit .env to add your GROQ_API_KEY
   ```

3. **Start the infrastructure**
   ```bash
   docker compose up -d
   ```

4. **Install dependencies and bootstrap**
   ```bash
   make install-dev
   # Bootstrap commands will be run in later phases
   ```

## 🛠️ Tech Stack

- **Vector Database**: Qdrant
- **Graph Database**: Neo4j Community
- **Embeddings**: BGE-M3 (via Ollama local)
- **LLM**: Groq API (Llama 3.1 70B)
- **Orchestration**: LangGraph & LlamaIndex
- **API**: FastAPI, Celery, Redis
- **Evaluation**: RAGAS & DeepEval
