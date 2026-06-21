# Architecture

LexRAG uses a modern Agentic RAG architecture:

1. **Ingestion Layer**: Supports unstructured multi-modal ingestion (PDFs, docx, etc.) with fallbacks for OCR. Uses Docling and Unstructured, feeding into a chunking pipeline (Hierarchical + Semantic).
2. **Indexing Layer**: Chunks are embedded using BGE-M3 (both dense vectors and BM25 sparse vectors). Entities and relations are extracted using spaCy and stored in Neo4j for Graph RAG.
3. **Retrieval Layer**: Reciprocal Rank Fusion (RRF) combines sparse and dense retrieval from Qdrant. A cross-encoder reranker (BGE Reranker v2) reranks the retrieved results.
4. **Graph Layer**: Queries trigger NER, and entities are queried against Neo4j to retrieve multi-hop context.
5. **Orchestration**: A LangGraph state machine orchestrates query analysis, retrieval, generation, and validation loops.
6. **API & Workers**: FastAPI serves endpoints, while Celery handles background document ingestion.
7. **Observability**: OpenTelemetry tracing via Arize Phoenix and metrics via Prometheus.
