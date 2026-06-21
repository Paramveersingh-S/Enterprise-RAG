# Enterprise RAG System — Deep Build Prompt
### Instruction file for AI-assisted end-to-end project construction

---

## MASTER CONTEXT

You are an expert AI/ML engineer specializing in production-grade Retrieval-Augmented Generation systems. Your task is to build a complete, enterprise-level RAG system from scratch using only free and open-source tools. The system is called **"LexRAG — Enterprise Legal Document Intelligence Platform"** and will serve as a showcase portfolio project demonstrating deep mastery of every layer of the modern RAG stack.

Every section you build must meet these standards:
- **Production quality**: Code must be clean, typed, documented, and testable — not tutorial-quality
- **Enterprise patterns**: Logging, error handling, retry logic, config management, async support throughout
- **Depth over breadth**: Implement each feature completely before moving to the next
- **Accuracy**: Do not hallucinate library APIs. If unsure, use the exact documented interface
- **Explainability**: Add detailed inline comments explaining *why* each architectural decision was made

The tech stack is fixed:
- **Document parsing**: Unstructured.io + Docling + PyMuPDF
- **Chunking**: LlamaIndex (hierarchical + semantic)
- **Embeddings**: BGE-M3 via Ollama (local, free)
- **Vector DB**: Qdrant (Docker, hybrid search)
- **Hybrid retrieval**: Qdrant RRF (BM25 + dense)
- **Re-ranking**: BGE Reranker v2 (local cross-encoder)
- **Graph layer**: Neo4j Community Edition + spaCy NER
- **Orchestration**: LangGraph + LlamaIndex
- **LLM**: Groq API (Llama 3.1 70B) + Ollama fallback
- **Evaluation**: RAGAS + DeepEval
- **Observability**: Arize Phoenix + LangSmith
- **API**: FastAPI + Redis + Celery
- **Deployment**: Docker Compose

---

## HOW TO USE THIS PROMPT FILE

Work through each `## PHASE` section sequentially. Each phase contains:
- **Objective**: What must be built
- **Acceptance criteria**: What "done" looks like — do not move to the next phase until all criteria are met
- **Implementation instructions**: Exact file names, structure, patterns, and code requirements
- **Test command**: How to verify the phase works correctly

Before starting each phase, re-read the master context above.

---

## PHASE 01 — Project Scaffolding & Configuration

### Objective
Create the full project directory structure, dependency management, configuration system, and base utilities that all subsequent phases will depend on.

### Acceptance criteria
- [ ] Project installs cleanly with `pip install -e .` in a fresh virtual environment
- [ ] All config values come from environment variables with Pydantic validation — no hardcoded secrets
- [ ] Logging is structured JSON to stdout (not print statements)
- [ ] Docker Compose file starts Qdrant, Neo4j, Redis, and Arize Phoenix with a single command
- [ ] A `make` command exists for every major operation

### Directory structure to create
```
lexrag/
├── pyproject.toml
├── Makefile
├── .env.example
├── .env                          # gitignored
├── docker-compose.yml
├── docker-compose.dev.yml
├── README.md
├── docs/
│   └── architecture.md
├── lexrag/
│   ├── __init__.py
│   ├── config.py                 # Pydantic Settings — all env vars
│   ├── logger.py                 # Structured JSON logger
│   ├── exceptions.py             # Custom exception hierarchy
│   └── utils/
│       ├── __init__.py
│       ├── retry.py              # Exponential backoff decorator
│       ├── timing.py             # Latency timing context manager
│       └── hashing.py            # Deterministic content hashing
├── ingestion/
│   ├── __init__.py
│   ├── parsers/
│   ├── chunkers/
│   └── pipeline.py
├── embeddings/
│   ├── __init__.py
│   └── encoder.py
├── retrieval/
│   ├── __init__.py
│   ├── vector_store.py
│   ├── hybrid.py
│   └── reranker.py
├── graph/
│   ├── __init__.py
│   ├── extractor.py
│   └── store.py
├── generation/
│   ├── __init__.py
│   ├── llm.py
│   └── prompts.py
├── orchestration/
│   ├── __init__.py
│   ├── graph_workflow.py
│   └── nodes.py
├── evaluation/
│   ├── __init__.py
│   ├── ragas_eval.py
│   └── deepeval_tests.py
├── api/
│   ├── __init__.py
│   ├── main.py
│   ├── routers/
│   │   ├── ingest.py
│   │   ├── query.py
│   │   └── health.py
│   ├── schemas.py
│   ├── middleware.py
│   └── dependencies.py
├── workers/
│   ├── __init__.py
│   └── tasks.py
└── tests/
    ├── conftest.py
    ├── unit/
    ├── integration/
    └── fixtures/
```

### Implementation instructions

**`pyproject.toml`**: Use `[project]` with `requires-python = ">=3.11"`. Pin major versions for all dependencies. Include optional dependency groups: `[project.optional-dependencies]` with `dev`, `test`, and `eval` groups. The main dependencies must include: `unstructured[all-docs]`, `docling`, `pymupdf`, `llama-index`, `llama-index-vector-stores-qdrant`, `qdrant-client`, `langchain`, `langgraph`, `langchain-groq`, `neo4j`, `spacy`, `sentence-transformers`, `fastapi`, `uvicorn[standard]`, `celery[redis]`, `redis`, `pydantic-settings`, `python-multipart`, `httpx`, `tenacity`, `structlog`, `ragas`, `deepeval`, `arize-phoenix`.

**`lexrag/config.py`**: Use `pydantic_settings.BaseSettings`. Define a `Settings` class with field groups separated by comments: QDRANT settings (host, port, collection name, vector size), NEO4J settings (uri, username, password), REDIS settings (url), GROQ settings (api_key, model name, temperature, max_tokens), OLLAMA settings (base_url, embedding_model, reranker_model), LANGSMITH settings (api_key, project name), ARIZE settings (endpoint), APPLICATION settings (environment, log_level, max_file_size_mb, supported_extensions). Add a `model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")`. Export a singleton `settings = Settings()` at module bottom.

**`lexrag/logger.py`**: Use `structlog` configured for JSON output in production, colored console output in development. Logger must automatically include: timestamp (ISO 8601), log level, module name, function name, line number. Export a `get_logger(name)` function. Every log call must be able to accept arbitrary keyword arguments as structured fields (e.g. `log.info("chunk created", chunk_id=..., token_count=...)`).

**`lexrag/exceptions.py`**: Define a base `LexRAGError(Exception)` with a `message`, `code`, and optional `details` dict. Subclass it for: `IngestionError`, `ParsingError`, `ChunkingError`, `EmbeddingError`, `RetrievalError`, `GraphError`, `GenerationError`, `RateLimitError`. Each exception must have a unique error code string (e.g. `"LEXRAG_PARSE_001"`).

**`docker-compose.yml`**: Define services: `qdrant` (image: `qdrant/qdrant:latest`, ports: 6333/6334, volume for persistence), `neo4j` (image: `neo4j:5-community`, ports: 7474/7687, env: `NEO4J_AUTH=neo4j/lexragpassword`, volume for data/logs), `redis` (image: `redis:7-alpine`, port: 6379), `phoenix` (image: `arizephoenix/phoenix:latest`, port: 6006). All services must have health checks. All must be on a shared `lexrag-network` bridge network.

**`Makefile`**: Define targets: `install`, `install-dev`, `format` (ruff + black), `lint` (ruff check + mypy), `test`, `test-unit`, `test-integration`, `eval`, `ingest` (runs ingestion pipeline on a sample), `serve` (uvicorn with reload), `docker-up`, `docker-down`, `docker-logs`, `clean`.

**`.env.example`**: Document every variable with a comment explaining what it controls and what the valid values are.

### Test command
```bash
make docker-up
python -c "from lexrag.config import settings; print(settings.model_dump())"
python -c "from lexrag.logger import get_logger; log = get_logger(__name__); log.info('scaffold ok', phase=1)"
```

---

## PHASE 02 — Document Ingestion & Parsing

### Objective
Build a robust, multi-format document ingestion pipeline that handles PDFs (native and scanned), Word documents, HTML, Markdown, Excel, and PowerPoint. Output must be clean, structured text with rich metadata. This is the most critical phase — bad parsing poisons everything downstream.

### Acceptance criteria
- [ ] Parses native PDFs preserving headers, paragraphs, tables, and lists
- [ ] Runs OCR on scanned PDFs via Tesseract and produces clean text
- [ ] Extracts tables from PDFs as structured Markdown, not garbled text
- [ ] Parses DOCX, PPTX, XLSX, HTML, and Markdown formats
- [ ] Every parsed document has a complete metadata dict attached
- [ ] Deduplication: identical content (same SHA-256 hash) is skipped silently
- [ ] All parsing errors are caught, logged with full context, and stored in a failure log — they do not crash the pipeline
- [ ] Processes a 100-page PDF in under 30 seconds

### Implementation instructions

**`ingestion/parsers/base.py`**: Define an abstract base class `BaseParser` with: `can_parse(file_path: Path) -> bool`, `parse(file_path: Path) -> ParsedDocument`. Define a `ParsedDocument` dataclass with fields: `content: str`, `metadata: DocumentMetadata`, `tables: list[Table]`, `images: list[ImageRef]`, `parse_errors: list[str]`. Define `DocumentMetadata` with: `source_path: str`, `filename: str`, `file_type: str`, `file_size_bytes: int`, `page_count: int`, `title: str | None`, `author: str | None`, `created_at: datetime | None`, `ingested_at: datetime`, `content_hash: str`, `language: str`, `word_count: int`.

**`ingestion/parsers/pdf_parser.py`**: Implement `PDFParser(BaseParser)`. Strategy: first attempt native text extraction with PyMuPDF (`fitz`). If a page has fewer than 50 characters of extractable text, fall back to OCR via Tesseract on a rasterized page image. For tables, use Docling's `DocumentConverter` which handles PDF tables with high accuracy. Preserve page numbers as metadata per text block. Log which pages used OCR vs native extraction. Handle password-protected PDFs gracefully (log and skip). Extract PDF metadata (title, author, subject) from the document info dict.

**`ingestion/parsers/docling_parser.py`**: Use `docling.DocumentConverter` as the primary parser for complex PDFs with mixed layouts. Configure it with `PdfFormatOption` to enable table structure recognition and formula recognition. Convert docling's output format to `ParsedDocument`. This parser should be preferred over the raw PDF parser for documents with complex layouts (research papers, legal contracts, financial reports).

**`ingestion/parsers/office_parser.py`**: Use `unstructured.partition.auto.partition()` for DOCX, PPTX, and XLSX. Map Unstructured element types to structured output: `Title` elements become section headers (prepend `##`), `Table` elements get converted to Markdown table syntax, `ListItem` elements get proper bullet formatting. Preserve slide numbers for PPTX. Extract sheet names for XLSX.

**`ingestion/parsers/web_parser.py`**: Handle HTML and Markdown. For HTML: use `unstructured.partition.html.partition_html()`, strip navigation/footer/sidebar elements by their common class names and tags, preserve article/main content. For Markdown: parse directly, preserve code blocks with language tags.

**`ingestion/parsers/registry.py`**: Implement a `ParserRegistry` class that holds a list of parsers and selects the right one via `can_parse()`. Add a method `get_parser(file_path: Path) -> BaseParser`. Raise `ParsingError` if no parser supports the file type.

**`ingestion/pipeline.py`**: Implement `IngestionPipeline` class. Constructor takes config and initializes all parsers. Main method: `async def ingest_file(file_path: Path) -> IngestionResult`. Steps: (1) compute SHA-256 hash of file, check Redis for duplicate by hash key `doc:hash:{hash}`, skip if exists; (2) select parser from registry; (3) parse document with error catching; (4) store hash in Redis with document ID as value; (5) log full metadata to structlog; (6) return `IngestionResult` with parsed document and status. Also implement `async def ingest_directory(dir_path: Path, recursive: bool = True)` that discovers all supported files and runs `ingest_file` concurrently with `asyncio.gather` (max 4 concurrent).

**`ingestion/pipeline.py` failure log**: Maintain a SQLite database (`lexrag_failures.db`) with a `parse_failures` table: `id`, `file_path`, `error_code`, `error_message`, `stack_trace`, `attempted_at`. Use `aiosqlite` for async writes. This is the enterprise pattern — failed ingestions are never silently dropped.

### Test command
```bash
# Download a test PDF (legal contract or research paper)
python -m lexrag.ingestion.pipeline --file tests/fixtures/sample_legal.pdf --verbose
# Check output: should print full ParsedDocument with metadata
pytest tests/unit/test_parsers.py -v
```

---

## PHASE 03 — Chunking Strategy

### Objective
Split parsed documents into semantically meaningful chunks using a hierarchical parent-child strategy. Chunks must carry rich metadata for later filtering. This phase directly controls retrieval accuracy — every decision here has downstream consequences.

### Acceptance criteria
- [ ] Implements hierarchical chunking: large parent chunks (512 tokens) contain smaller child chunks (128 tokens)
- [ ] Child chunks embed parent context in their metadata (not their text — to avoid redundancy)
- [ ] Semantic chunking available as an option: splits on embedding-similarity drops, not character count
- [ ] Sentence-window chunking available: stores ±3 sentences around each sentence for context
- [ ] Every chunk carries: chunk_id (UUID), document_id, parent_chunk_id (if child), chunk_index, total_chunks, page_numbers, section_header, token_count, character_count, chunking_strategy used
- [ ] Chunks are serializable to JSON for storage and inspection
- [ ] Token counting uses `tiktoken` (cl100k_base) for accuracy — not character approximations

### Implementation instructions

**`ingestion/chunkers/base.py`**: Define `BaseChunker` abstract class with `chunk(document: ParsedDocument) -> list[Chunk]`. Define `Chunk` dataclass: `chunk_id: str`, `document_id: str`, `parent_chunk_id: str | None`, `text: str`, `metadata: ChunkMetadata`. Define `ChunkMetadata`: `chunk_index: int`, `total_chunks: int`, `page_numbers: list[int]`, `section_header: str | None`, `token_count: int`, `character_count: int`, `strategy: str`, `overlap_with_previous: bool`, `document_metadata: DocumentMetadata`.

**`ingestion/chunkers/hierarchical.py`**: Implement `HierarchicalChunker(BaseChunker)`. Two-pass approach: Pass 1 creates parent chunks using `RecursiveCharacterTextSplitter` with `chunk_size=2048` tokens, `chunk_overlap=200`. Pass 2 splits each parent into child chunks with `chunk_size=512` tokens, `chunk_overlap=50`. Child chunks store `parent_chunk_id` and `parent_text` in metadata (NOT in their `text` field). At retrieval time, child chunks will be used for matching but parent text will be passed to the LLM — this is the "small-to-big" retrieval pattern. Use LlamaIndex's `HierarchicalNodeParser` to implement this properly.

**`ingestion/chunkers/semantic.py`**: Implement `SemanticChunker(BaseChunker)`. Embed each sentence using the configured embedding model. Compute cosine similarity between adjacent sentence embeddings. Insert a chunk boundary whenever similarity drops below a configurable threshold (default: 0.75). This ensures chunks are semantically coherent even if they vary in token count. Enforce min/max token limits (min: 64, max: 1024) to prevent micro-chunks and giant chunks.

**`ingestion/chunkers/sentence_window.py`**: Implement `SentenceWindowChunker(BaseChunker)`. Split document into individual sentences using spaCy `en_core_web_sm`. For each sentence, create a chunk whose `text` is that single sentence but whose `metadata.window_text` contains the surrounding `window_size` sentences (default: 3 on each side). This window text is what gets passed to the LLM — the individual sentence is what gets embedded and searched. Use LlamaIndex's `SentenceWindowNodeParser`.

**`ingestion/chunkers/factory.py`**: Implement `ChunkerFactory` with a `create(strategy: str) -> BaseChunker` class method. Strategy options: `"hierarchical"` (default for legal docs), `"semantic"` (for research papers), `"sentence_window"` (for QA over dense text), `"fixed"` (fallback).

**Section header detection**: In all chunkers, detect section headers by checking if a text segment matches patterns like `\n[A-Z][A-Z ]{5,}\n`, numbered headings (`1.`, `1.1`), or Markdown headings (`##`). Attach the most recent detected header to each chunk's metadata.

**`ingestion/chunkers/token_counter.py`**: Implement `TokenCounter` with a `count(text: str) -> int` method using `tiktoken.get_encoding("cl100k_base")`. Cache the encoding object at module level — instantiating it repeatedly is expensive. Add a `split_to_fit(text: str, max_tokens: int) -> list[str]` utility.

### Test command
```bash
python -m pytest tests/unit/test_chunkers.py -v
# Also run the visual inspection script:
python scripts/inspect_chunks.py --file tests/fixtures/sample_legal.pdf --strategy hierarchical
# Should print a tree of parent → child chunks with token counts
```

---

## PHASE 04 — Embedding Pipeline

### Objective
Encode all chunks into dense vector embeddings using BGE-M3 running locally via Ollama. Build a production-grade encoding pipeline with batching, caching, and error recovery.

### Acceptance criteria
- [ ] Uses BGE-M3 model via Ollama REST API — no cloud API calls, zero cost
- [ ] Batches chunks in groups of 32 for efficient encoding (single chunks are slow)
- [ ] Caches embeddings in Redis keyed by chunk content hash — avoids re-encoding unchanged content
- [ ] Handles Ollama connection failures with exponential backoff (3 retries, 1s/2s/4s delays)
- [ ] Returns embeddings as `numpy.ndarray` with shape `(n_chunks, 1024)` (BGE-M3 dimension)
- [ ] Encodes 1000 chunks in under 5 minutes on CPU
- [ ] Embedding dimension is configurable via settings — changing to a different model doesn't require code changes

### Implementation instructions

**`embeddings/encoder.py`**: Implement `BGEEncoder` class. Constructor: `__init__(self, ollama_url: str, model: str = "bge-m3", cache: Redis | None = None)`. Method `async def encode(texts: list[str]) -> np.ndarray`. Internally: (1) check Redis cache for each text hash, separate cache-hit texts from cache-miss texts; (2) batch cache-miss texts into groups of 32; (3) for each batch, call `POST {ollama_url}/api/embed` with `{"model": self.model, "input": batch}`; (4) parse response `embeddings` field (list of float lists); (5) store new embeddings in Redis with TTL of 7 days; (6) reconstruct the full embedding array in original order; (7) return as `np.ndarray(dtype=np.float32)`.

**Ollama API call details**: The Ollama `/api/embed` endpoint accepts `{"model": "...", "input": ["text1", "text2", ...]}` and returns `{"embeddings": [[...], [...]]}`. Use `httpx.AsyncClient` with a timeout of 120 seconds. The client should be created once and reused (connection pooling).

**`embeddings/encoder.py` — retry logic**: Decorate the `_call_ollama_batch()` private method with `@tenacity.retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(httpx.ConnectError))`. Log each retry attempt with structlog including the batch index and error message.

**`embeddings/encoder.py` — cache key**: `embedding_cache:{sha256(text)[:16]}`. Do NOT use the full SHA-256 as the key — it wastes Redis memory. The first 16 hex characters give 64 bits of collision resistance, more than sufficient.

**`embeddings/encoder.py` — normalization**: BGE-M3 embeddings should be L2-normalized before storing. This makes cosine similarity equivalent to dot product, which Qdrant can compute faster. Normalize with `embedding / np.linalg.norm(embedding)`.

**`embeddings/encoder.py` — dimension check**: After encoding, assert that the embedding dimension matches `settings.embedding_dim`. If it doesn't, raise `EmbeddingError` with details — this catches model mismatches early.

**`embeddings/dual_encoder.py`**: BGE-M3 supports both dense and sparse (BM25-style) embeddings in a single model pass. Implement `DualEncoder` that extracts both `dense_vecs` and `sparse_vecs` from BGE-M3's output. The sparse vectors are needed for hybrid search in Qdrant. This is a key advanced feature — most implementations only use the dense vectors and miss the sparse signal entirely.

### Test command
```bash
# First ensure Ollama is running with BGE-M3 pulled:
ollama pull bge-m3
# Then run:
python -c "
import asyncio
from lexrag.embeddings.encoder import BGEEncoder
enc = BGEEncoder(ollama_url='http://localhost:11434')
vecs = asyncio.run(enc.encode(['enterprise legal document', 'force majeure clause']))
print('Shape:', vecs.shape)  # Should be (2, 1024)
print('Norm:', (vecs**2).sum(axis=1)**0.5)  # Should be [1.0, 1.0] if normalized
"
```

---

## PHASE 05 — Vector Store & Hybrid Retrieval

### Objective
Store all chunk embeddings in Qdrant and implement hybrid retrieval combining dense (BGE-M3) and sparse (BM25) search with Reciprocal Rank Fusion. This is the core retrieval engine.

### Acceptance criteria
- [ ] Qdrant collection created with HNSW index, cosine distance, named vectors for dense and sparse
- [ ] Upserts handle duplicates gracefully (same chunk_id = update, not duplicate)
- [ ] Dense retrieval returns top-k by cosine similarity
- [ ] Sparse retrieval returns top-k by BM25 score
- [ ] Hybrid search combines both with Reciprocal Rank Fusion (k=60)
- [ ] Metadata filtering works: filter by document_id, section_header, page_number range
- [ ] Retrieval latency under 200ms for a collection of 100,000 chunks
- [ ] Search results include relevance scores, chunk text, and full metadata

### Implementation instructions

**`retrieval/vector_store.py`**: Implement `QdrantVectorStore` class. Constructor initializes `QdrantClient` (async client). Method `async def create_collection(collection_name: str)`: create collection with `VectorsConfig` containing two named vectors: `"dense"` (size=1024, distance=Distance.COSINE) and `"sparse"` (using `SparseVectorParams`). Set HNSW config: `m=16, ef_construct=100`. Set `on_disk_payload=True` for production (saves RAM). Add `PayloadSchemaIndex` for fields: `document_id` (keyword), `section_header` (keyword), `page_numbers` (integer), `chunk_index` (integer).

**`retrieval/vector_store.py` — upsert**: Method `async def upsert_chunks(chunks: list[Chunk], dense_embeddings: np.ndarray, sparse_embeddings: list[dict])`. Construct `PointStruct` for each chunk with: `id` = UUID from `chunk_id`, `vector = {"dense": dense_vec.tolist(), "sparse": {"indices": sparse_indices, "values": sparse_values}}`, `payload` = full chunk metadata dict. Use `client.upsert(collection_name, points, wait=True)` in batches of 100.

**`retrieval/vector_store.py` — delete**: Method `async def delete_document(document_id: str)`. Use Qdrant filter to delete all points where `payload.document_id == document_id`. Essential for re-ingestion workflows.

**`retrieval/hybrid.py`**: Implement `HybridRetriever` class. Core method: `async def retrieve(query: str, top_k: int = 20, filter: Filter | None = None) -> list[RetrievalResult]`. Steps: (1) encode query to dense vector and sparse vector using `DualEncoder`; (2) run Qdrant `query_points` with `prefetch=[Prefetch(query=dense_vec, using="dense", limit=top_k*2), Prefetch(query=sparse_vec, using="sparse", limit=top_k*2)]` and `query=FusionQuery(fusion=Fusion.RRF)`; (3) map raw Qdrant results to `RetrievalResult` objects.

**`retrieval/hybrid.py` — RetrievalResult**: Define dataclass: `chunk_id: str`, `document_id: str`, `text: str`, `parent_text: str | None`, `score: float`, `rank: int`, `metadata: ChunkMetadata`, `retrieval_method: str` (e.g. "hybrid_rrf").

**`retrieval/hybrid.py` — query expansion**: Before retrieval, optionally expand the query using the LLM to generate 2-3 alternative phrasings. This is the "Multi-Query Retrieval" technique. Run all queries in parallel, collect all results, deduplicate by chunk_id, keep the highest score for each. Controlled by a `use_query_expansion: bool` flag.

**`retrieval/hybrid.py` — MMR (Maximal Marginal Relevance)**: After initial retrieval, optionally apply MMR to balance relevance vs diversity. For each next result, pick the one that maximizes `λ * similarity_to_query - (1-λ) * max_similarity_to_selected`. This prevents the LLM from receiving 5 nearly-identical chunks. Controlled by `use_mmr: bool` and `mmr_lambda: float = 0.5`.

### Test command
```bash
# Start Qdrant first: make docker-up
python scripts/test_retrieval.py --query "force majeure clause in commercial contracts"
# Should print: top-5 chunks with scores, retrieval method, and latency
pytest tests/integration/test_vector_store.py -v
```

---

## PHASE 06 — Re-ranking

### Objective
Implement a cross-encoder re-ranker using BGE Reranker v2 that takes the top-20 retrieved chunks and re-scores them with the full query-document interaction, producing a top-5 set with dramatically improved precision.

### Acceptance criteria
- [ ] Uses BGE Reranker v2 (`BAAI/bge-reranker-v2-m3`) as a cross-encoder
- [ ] Runs entirely locally via SentenceTransformers — zero cloud cost
- [ ] Takes (query, chunk) pairs as input, returns relevance scores in [0, 1]
- [ ] Re-ranks in batches for efficiency (max 16 pairs per batch)
- [ ] If parent chunks exist, re-ranks on parent text (more context = better scoring)
- [ ] Returns ranked list with original retrieval rank AND reranker rank for comparison
- [ ] Logs rank change statistics (e.g. average position shift) for monitoring

### Implementation instructions

**`retrieval/reranker.py`**: Implement `CrossEncoderReranker` class. Constructor loads `CrossEncoder("BAAI/bge-reranker-v2-m3", max_length=512)` from SentenceTransformers. Model is loaded once at startup — add a singleton pattern. Method `def rerank(query: str, results: list[RetrievalResult], top_n: int = 5) -> list[RetrievalResult]`: build pairs list as `[(query, r.parent_text or r.text) for r in results]`; call `self.model.predict(pairs, batch_size=16, show_progress_bar=False)`; attach scores to results; sort descending by score; take top_n; set `reranker_rank` field on each; log average rank change.

**`retrieval/reranker.py` — scoring**: Normalize raw cross-encoder logits to [0, 1] using sigmoid: `score = 1 / (1 + exp(-logit))`. This gives an interpretable "relevance probability" score.

**`retrieval/reranker.py` — threshold filtering**: After re-ranking, filter out any result with a score below `min_relevance_threshold` (default: 0.3). If all results fall below threshold, return the top-1 anyway (never return empty) but log a warning. This prevents the LLM from hallucinating when no relevant context exists.

**`retrieval/reranker.py` — extended logging**: For every re-rank call, log a `structlog` record with: `query`, `input_count`, `output_count`, `top_score`, `bottom_score`, `mean_score`, `latency_ms`, `rank_correlation` (Spearman between original and reranked orders — a measure of how much the reranker changed things).

**`retrieval/pipeline.py`**: Compose `HybridRetriever` + `CrossEncoderReranker` into a `RetrievalPipeline` class with a single method `async def retrieve_and_rerank(query: str, top_k: int = 20, top_n: int = 5, filter: Filter | None = None) -> list[RetrievalResult]`.

### Test command
```bash
python scripts/compare_retrieval.py --query "termination for convenience clause"
# Should print a side-by-side table: chunk text | retrieval_rank | reranker_rank | score
# Verify that reranker meaningfully changes the order
```

---

## PHASE 07 — Graph RAG Layer

### Objective
Extract entities and relationships from all ingested documents and store them in a Neo4j knowledge graph. Enable multi-hop graph traversal as a retrieval strategy complementary to vector search. This is the enterprise-differentiating feature.

### Acceptance criteria
- [ ] Extracts named entities: PERSON, ORG, DATE, LAW, MONEY, GPE, CONTRACT_PARTY using spaCy
- [ ] Extracts relationships between entities from the same sentence using dependency parsing
- [ ] Stores entities as Neo4j nodes with properties: name, entity_type, document_ids, first_seen
- [ ] Stores relationships as Neo4j edges with properties: relation_type, source_document, context_sentence
- [ ] Graph traversal: given a query entity, return all connected entities up to 2 hops
- [ ] Hybrid graph+vector: graph results get added to vector results, passed together to re-ranker
- [ ] Neo4j queries use parameterized Cypher — no string formatting (SQL injection equivalent)

### Implementation instructions

**`graph/extractor.py`**: Implement `EntityRelationExtractor`. Load `spacy.load("en_core_web_lg")` — the large model for better NER accuracy (download with `python -m spacy download en_core_web_lg`). Method `extract(chunks: list[Chunk]) -> GraphExtracts`. For each chunk: (1) run `nlp(chunk.text)` to get spaCy doc; (2) extract named entities as `Entity(text=ent.text, label=ent.label_, start_char=ent.start_char, end_char=ent.end_char, sentence=ent.sent.text)`; (3) for each sentence, find pairs of entities that co-occur and extract the dependency path between them as the relation type (e.g. "SUBJECT_OF", "PARTY_TO", "SIGNED_BY"). Return `GraphExtracts(entities: list[Entity], relations: list[Relation], source_chunk_id: str)`.

**`graph/extractor.py` — legal NER extension**: Legal documents have entity types not in standard NER. Add a rule-based component using `spacy.pipeline.EntityRuler` to recognize: contract parties (patterns: "Party A", "the Client", "the Vendor"), legal references ("Section 12.3", "Exhibit A", "Schedule 1"), monetary amounts with context ("damages of $X", "$X per annum"), and date references with context. These patterns should be loaded from a JSON file `lexrag/graph/legal_patterns.json` for easy extension.

**`graph/store.py`**: Implement `Neo4jGraphStore` using the official `neo4j` Python driver (async). Method `async def upsert_entity(entity: Entity)`: use `MERGE (e:Entity {name: $name, type: $type}) ON CREATE SET e.first_seen = $now, e.document_ids = [$doc_id] ON MATCH SET e.document_ids = CASE WHEN $doc_id IN e.document_ids THEN e.document_ids ELSE e.document_ids + [$doc_id] END`. Method `async def upsert_relation(relation: Relation)`: `MATCH (a:Entity {name: $from_name}), (b:Entity {name: $to_name}) MERGE (a)-[r:RELATED_TO {type: $rel_type}]->(b) ON CREATE SET r.contexts = [$context] ON MATCH SET r.contexts = r.contexts + [$context]`.

**`graph/store.py` — query method**: `async def get_entity_neighbors(entity_name: str, hops: int = 2) -> GraphQueryResult`. Cypher: `MATCH path = (e:Entity {name: $name})-[*1..{hops}]-(neighbor) RETURN path, neighbor`. Return a `GraphQueryResult` with nodes, edges, and extracted text snippets from `r.contexts`.

**`graph/store.py` — entity disambiguation**: Common issue: "IBM", "International Business Machines", and "IBM Corp." are the same entity. After extraction, run a simple normalization pass: lowercase comparison, remove common suffixes (Inc., Corp., Ltd., LLC.). Use `rapidfuzz` for fuzzy matching: if two entity names have similarity > 0.92, treat them as the same entity and merge their Neo4j nodes.

**Integration in retrieval**: In `orchestration/graph_workflow.py`, add a graph retrieval node that: (1) runs NER on the query to identify mentioned entities; (2) queries Neo4j for each found entity and its neighbors; (3) collects the `context_sentence` strings from the edges; (4) creates synthetic `RetrievalResult` objects from these context strings; (5) merges them with the vector retrieval results before re-ranking.

### Test command
```bash
python scripts/explore_graph.py --entity "IBM Corporation"
# Should print: all connected entities, relationship types, and source sentences
# Also verify via Neo4j Browser at http://localhost:7474
```

---

## PHASE 08 — LangGraph Orchestration

### Objective
Wire all components into a stateful LangGraph workflow that implements Agentic RAG: the LLM decides when to retrieve, whether to refine the query, and whether to use graph or vector search. Build a complete multi-step reasoning pipeline.

### Acceptance criteria
- [ ] LangGraph state is fully typed with a `TypedDict` state schema
- [ ] Graph has at minimum 6 nodes: query_analysis, retrieve, rerank, graph_retrieve, generate, validate
- [ ] Conditional edges: if query is ambiguous → query_refinement node; if answer fails validation → retrieve again with refined query
- [ ] Maximum 3 retrieval iterations (prevents infinite loops)
- [ ] Every node logs its input state and output state with structlog
- [ ] The complete workflow is traceable in LangSmith
- [ ] Can handle multi-turn conversation: state includes `chat_history: list[BaseMessage]`

### Implementation instructions

**`orchestration/state.py`**: Define `RAGState(TypedDict)` with fields: `query: str`, `original_query: str`, `chat_history: Annotated[list[BaseMessage], add_messages]`, `retrieved_chunks: list[RetrievalResult]`, `graph_results: list[RetrievalResult]`, `reranked_chunks: list[RetrievalResult]`, `generation: str | None`, `citations: list[Citation]`, `confidence_score: float`, `retrieval_iterations: int`, `query_was_refined: bool`, `refined_query: str | None`, `error: str | None`, `metadata: dict`.

**`orchestration/nodes.py`**: Implement each node as an async function `async def node_name(state: RAGState) -> dict`. Nodes must only return the keys they modify — LangGraph merges these into the state.

- `query_analysis_node`: Call LLM with a system prompt to classify the query: is it a factual lookup, a multi-hop reasoning question, a document comparison, or a conversational follow-up? Set `state["query_type"]` and optionally `state["refined_query"]`. If the query references "the contract", "the document", "it" — resolve the reference using chat history.

- `vector_retrieve_node`: Call `RetrievalPipeline.retrieve_and_rerank()`. Set `state["retrieved_chunks"]`. Increment `state["retrieval_iterations"]`.

- `graph_retrieve_node`: Run NER on query, query Neo4j for entity context. Set `state["graph_results"]`. If no entities found in query, return immediately without Neo4j call.

- `merge_and_rerank_node`: Combine `retrieved_chunks` + `graph_results`, deduplicate by chunk_id, run `CrossEncoderReranker` on the combined set. Set `state["reranked_chunks"]`.

- `generate_node`: Build the final prompt. Format retrieved chunks as numbered citations `[1] ... [2] ...`. Call Groq API with Llama 3.1 70B. Parse the response to extract: answer text and citation references `[1]`, `[2]`. Build `Citation` objects linking back to `reranked_chunks`. Set `state["generation"]` and `state["citations"]`.

- `validate_node`: Check the generated answer: (1) does it contain any `[N]` citations? (2) are the citations grounded in the retrieved text (basic overlap check)? (3) does the answer address the query (LLM-as-judge call with simple yes/no)? Set `state["confidence_score"]`. If score < 0.6 and `retrieval_iterations` < 3, return a signal to retry.

**`orchestration/graph_workflow.py`**: Build the LangGraph `StateGraph`. Add all nodes. Add edges: `START → query_analysis`. Conditional edge from `query_analysis`: if `query_type == "conversational"` → go straight to generate; else → vector_retrieve. After `vector_retrieve` → `graph_retrieve` → `merge_and_rerank` → `generate` → `validate`. Conditional edge from `validate`: if confidence < threshold AND iterations < 3 → `query_analysis` (with refined query); else → `END`. Compile the graph with `graph.compile(checkpointer=MemorySaver())`.

**`orchestration/prompts.py`**: Define all prompts as structured `ChatPromptTemplate` objects. Never hardcode prompt strings inline in nodes. Prompts needed: `QUERY_ANALYSIS_PROMPT`, `RAG_GENERATION_PROMPT`, `ANSWER_VALIDATION_PROMPT`, `QUERY_REFINEMENT_PROMPT`. The RAG generation prompt must instruct the model to: answer using ONLY the provided context, cite sources by number, say "I cannot find this in the provided documents" if context is insufficient — never hallucinate.

### Test command
```bash
python scripts/run_query.py --query "What are the termination conditions in the IBM services agreement?"
# Should print: the full state trace showing each node, retrieved chunks, and final answer with citations
# Check LangSmith dashboard for the full trace
```

---

## PHASE 09 — Evaluation Suite

### Objective
Build a comprehensive evaluation pipeline using RAGAS and DeepEval that measures retrieval quality, generation quality, and end-to-end performance. Generate a synthetic test dataset from your documents. This section is what proves your system works.

### Acceptance criteria
- [ ] Generates a synthetic QA test dataset with 50+ question-answer-context triplets from your documents
- [ ] Measures all 4 core RAGAS metrics: Context Precision, Context Recall, Faithfulness, Answer Relevancy
- [ ] DeepEval tests run as part of `pytest` and fail CI if metrics drop below thresholds
- [ ] Evaluation results are saved to JSON with timestamps for trend tracking
- [ ] A comparison script shows metric improvements as you refine the pipeline
- [ ] Hallucination rate (1 - Faithfulness) must be below 0.15

### Implementation instructions

**`evaluation/dataset_generator.py`**: Use RAGAS `TestsetGenerator` with your documents. `generator = TestsetGenerator.from_llm(generator_llm=groq_llm, critic_llm=groq_llm, embeddings=bge_embeddings)`. Call `generator.generate_with_llamaindex_docs(documents, test_size=50, distributions={simple: 0.5, reasoning: 0.3, multi_context: 0.2})`. Save to `tests/fixtures/eval_dataset.json`. The distribution ensures a mix of easy lookup questions, multi-hop reasoning questions, and questions requiring synthesis across multiple chunks.

**`evaluation/ragas_eval.py`**: Implement `RAGASEvaluator` class. Method `async def evaluate(dataset: EvalDataset) -> RAGASReport`. For each question in dataset: run the full LangGraph pipeline to get `answer` and `contexts`. Build a `Dataset` object for RAGAS with columns: `question`, `answer`, `contexts`, `ground_truth`. Run `evaluate(dataset, metrics=[context_precision, context_recall, faithfulness, answer_relevancy])`. Save results to `results/ragas_{timestamp}.json`. Print a summary table with mean scores per metric.

**`evaluation/deepeval_tests.py`**: Write DeepEval test cases using `pytest`. Each test is a `@pytest.mark.parametrize` over sample questions. Define `min_score` thresholds: Faithfulness ≥ 0.85, Answer Relevancy ≥ 0.80, Context Recall ≥ 0.75. Use `assert_test(test_case, [FaithfulnessMetric(threshold=0.85), AnswerRelevancyMetric(threshold=0.80)])`. These tests run in CI via `make eval`.

**`evaluation/metrics_tracker.py`**: Load all `results/ragas_*.json` files and compute metric trends over time. Print a trend table and flag any metric that has declined by more than 0.05 since the previous run.

**`evaluation/ablation.py`**: Script to compare different pipeline configurations: (1) vector-only retrieval; (2) hybrid retrieval; (3) hybrid + reranker; (4) hybrid + reranker + graph. Run RAGAS evaluation for each configuration on the same dataset. Print a comparison table. This is what you use to prove that each layer adds measurable value.

### Test command
```bash
make eval
# Should run full RAGAS evaluation and print metric scores
# Example target output:
# context_precision:  0.87
# context_recall:     0.81
# faithfulness:       0.89
# answer_relevancy:   0.84
```

---

## PHASE 10 — FastAPI Backend

### Objective
Expose the entire RAG system as a production-grade REST API with authentication, rate limiting, request validation, async processing, and comprehensive error handling.

### Acceptance criteria
- [ ] `POST /api/v1/ingest` accepts file upload, runs ingestion asynchronously via Celery, returns job ID
- [ ] `GET /api/v1/ingest/{job_id}` returns job status and progress
- [ ] `POST /api/v1/query` accepts query + optional filters, returns answer + citations + confidence score
- [ ] `GET /api/v1/documents` lists all ingested documents with metadata
- [ ] `DELETE /api/v1/documents/{document_id}` removes document from all stores (vector DB + graph)
- [ ] All endpoints require a Bearer token (static API key for the portfolio demo)
- [ ] Rate limiting: 10 queries/minute per API key via Redis
- [ ] Request/response validation via Pydantic models
- [ ] All errors return structured JSON with error code, message, and request ID
- [ ] OpenAPI docs auto-generated and accurate at `/docs`

### Implementation instructions

**`api/schemas.py`**: Define all Pydantic models. `IngestRequest`: `file_name: str`, `metadata: dict = {}`. `IngestResponse`: `job_id: str`, `status: str`, `message: str`. `QueryRequest`: `query: str = Field(min_length=5, max_length=2000)`, `top_k: int = Field(default=5, ge=1, le=20)`, `filters: dict = {}`, `use_graph: bool = True`, `session_id: str | None = None`. `QueryResponse`: `answer: str`, `citations: list[Citation]`, `confidence_score: float`, `retrieval_method: str`, `latency_ms: float`, `session_id: str`. `Citation`: `chunk_id: str`, `document_id: str`, `text: str`, `page_numbers: list[int]`, `relevance_score: float`. `ErrorResponse`: `error_code: str`, `message: str`, `request_id: str`, `timestamp: str`.

**`api/middleware.py`**: Implement three middleware components using Starlette's `BaseHTTPMiddleware`: (1) `RequestIDMiddleware`: generates a UUID for each request, attaches to request state and response header `X-Request-ID`; (2) `TimingMiddleware`: measures request duration, attaches to response header `X-Response-Time-Ms`; (3) `LoggingMiddleware`: logs every request with method, path, status code, latency, and request ID using structlog.

**`api/routers/query.py`**: The query endpoint must: (1) check Redis rate limiter using sliding window algorithm (lua script for atomicity); (2) validate API key from `Authorization: Bearer {key}` header; (3) get or create a LangGraph conversation thread for the session_id; (4) invoke the LangGraph workflow; (5) extract and format the response; (6) send a copy to Arize Phoenix for observability; (7) cache the response in Redis for 5 minutes for identical queries (keyed by hash of query + filters).

**`api/routers/ingest.py`**: The ingest endpoint: (1) save uploaded file to a temp directory; (2) send a Celery task `tasks.process_document.delay(file_path, metadata)`; (3) store job status in Redis as `job:{job_id}` with initial status "queued"; (4) return the job_id. The Celery task runs the full ingestion pipeline, updates job status in Redis at each step ("parsing", "chunking", "embedding", "indexing", "done"), and writes the final document ID back to the job record.

**`api/main.py`**: Create `FastAPI(title="LexRAG API", version="1.0.0", description="Enterprise Legal Document Intelligence")`. Include all routers with prefix `/api/v1`. Add middleware in correct order (request ID first, then logging, then timing). Add a startup event that: verifies Qdrant connection, verifies Neo4j connection, verifies Ollama is running, verifies Redis connection — fail fast if any dependency is unavailable. Add a `/health` endpoint that returns the status of each dependency.

**`workers/tasks.py`**: Use `celery.Celery` with Redis as both broker and result backend. Task `process_document(file_path: str, job_id: str, metadata: dict)`: run the full `IngestionPipeline` + chunking + embedding + vector store upsert + graph extraction. Update Redis job status at each step. Handle failures with `self.retry(countdown=60, max_retries=2)`.

### Test command
```bash
make serve
# In another terminal:
curl -X POST http://localhost:8000/api/v1/query \
  -H "Authorization: Bearer lexrag-dev-key" \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the liability caps in the services agreement?"}'
# Should return structured JSON with answer, citations, and confidence score
```

---

## PHASE 11 — Observability & Monitoring

### Objective
Instrument the entire pipeline with structured logging, distributed tracing, and metrics. Every RAG query should be fully reconstructable from logs — which chunks were retrieved, what scores they had, how the answer was generated.

### Acceptance criteria
- [ ] Every LangGraph node emits a span to Arize Phoenix
- [ ] Every Qdrant retrieval is traced with query, filter, results, and latency
- [ ] Every LLM call is traced with prompt tokens, completion tokens, cost estimate, and latency
- [ ] Groq API key usage is monitored — log remaining rate limit from response headers
- [ ] Prometheus metrics exported at `/metrics`: query_count, query_latency_histogram, retrieval_latency_histogram, rerank_latency_histogram, llm_latency_histogram, ingestion_success_count, ingestion_failure_count
- [ ] Grafana dashboard JSON provided (importable) with panels for all key metrics

### Implementation instructions

**`lexrag/tracing.py`**: Initialize Arize Phoenix with `px.launch_app()` in development. Configure OpenTelemetry tracer with `OTLPSpanExporter` pointing to Phoenix. Implement a `@traced(name: str)` decorator that wraps any async function in an OTEL span, attaches `span.set_attribute("input", str(args))` and `span.set_attribute("output", str(result)[:500])`. Apply this decorator to: `retrieve_and_rerank`, `rerank`, `generate_node`, `graph_retrieve_node`.

**LLM tracing**: Use LangSmith by setting environment variables `LANGCHAIN_TRACING_V2=true`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT`. LangGraph and LangChain calls are automatically traced — no code changes needed. Verify in LangSmith dashboard that you can see full traces with input/output for every node.

**`api/metrics.py`**: Use `prometheus_client`. Define: `QUERY_COUNT = Counter("lexrag_queries_total", "Total queries", ["status"])`, `QUERY_LATENCY = Histogram("lexrag_query_duration_seconds", "Query duration", buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0])`, similar histograms for retrieval, rerank, llm. Mount Prometheus metrics endpoint on FastAPI at `/metrics` using `make_asgi_app()`.

**Grafana dashboard**: Provide `monitoring/grafana_dashboard.json` that imports into Grafana (running in Docker). Dashboard panels: query rate (queries/min), p50/p95/p99 query latency, retrieval latency breakdown, LLM call latency, ingestion queue depth (from Celery), error rate.

### Test command
```bash
# Run several queries then open:
# Arize Phoenix: http://localhost:6006 — should show full spans
# Grafana: http://localhost:3000 — should show metrics panels
# LangSmith: https://smith.langchain.com — should show LangGraph traces
```

---

## PHASE 12 — Docker Compose & Deployment

### Objective
Containerize the complete system so it starts from zero with a single command. Write a `docker-compose.yml` that orchestrates all services. Create deployment documentation for free hosting on Hugging Face Spaces and Railway.

### Acceptance criteria
- [ ] `docker compose up --build` starts the entire system from scratch
- [ ] All services have health checks and proper startup ordering (`depends_on: condition: service_healthy`)
- [ ] The API is accessible at `http://localhost:8000`
- [ ] Qdrant, Neo4j, and Redis data persist across restarts via Docker volumes
- [ ] A `scripts/bootstrap.sh` script: pulls Ollama models, creates Qdrant collection, creates Neo4j constraints, loads sample documents
- [ ] The README has a one-command install and demo section

### Implementation instructions

**`docker-compose.yml` services**: Add the LexRAG API service: build from `Dockerfile`, `depends_on: qdrant, neo4j, redis` with `condition: service_healthy`, environment from `.env` file, ports 8000:8000. Add Celery worker service: same image, command `celery -A lexrag.workers.tasks worker --loglevel=info --concurrency=2`. Add Ollama service: image `ollama/ollama`, GPU support via `deploy.resources.reservations.devices` for NVIDIA (with CPU fallback). Add Prometheus: `prom/prometheus`, config from `monitoring/prometheus.yml`. Add Grafana: `grafana/grafana`, provisioned with dashboard JSON.

**`Dockerfile`**: Multi-stage build. Stage 1 `builder`: `python:3.11-slim`, install system deps (`tesseract-ocr`, `libgl1`, `poppler-utils`), copy `pyproject.toml`, run `pip install .`. Stage 2 `runtime`: copy installed packages from builder, copy source code, set `CMD ["uvicorn", "lexrag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]`. Use non-root user for security.

**`scripts/bootstrap.sh`**: (1) wait for Ollama to be healthy; (2) `ollama pull bge-m3`; (3) `ollama pull llama3.2` (for local LLM fallback); (4) `python -m lexrag.retrieval.vector_store create-collection`; (5) `python -m lexrag.graph.store create-constraints`; (6) `python -m lexrag.ingestion.pipeline --directory tests/fixtures/sample_docs/`. Print progress with emoji checkmarks.

**`README.md`**: Must include: project overview with architecture diagram (Mermaid), quick start (3 commands: clone, copy .env.example, docker compose up), API usage examples with `curl`, evaluation results table showing RAGAS metrics, project structure explanation, and a section "What I learned building this" — important for portfolio framing.

### Test command
```bash
# From a clean machine:
git clone <your-repo>
cd lexrag
cp .env.example .env
docker compose up --build
# Wait ~3 minutes for models to download
bash scripts/bootstrap.sh
curl http://localhost:8000/health
# Should return {"status": "healthy", "services": {"qdrant": "ok", "neo4j": "ok", "redis": "ok", "ollama": "ok"}}
```

---

## PHASE 13 — Portfolio Presentation Layer

### Objective
Build a minimal, impressive Streamlit demo UI that showcases every feature of the system for portfolio viewers and interviewers. This is the "wow factor" layer.

### Acceptance criteria
- [ ] Chat interface showing the full conversation history
- [ ] Side panel shows retrieved chunks with relevance scores for each query
- [ ] Citation viewer: click a citation number to see the exact source chunk and page number
- [ ] Graph visualization: shows the entity graph for the current query (using pyvis)
- [ ] Metrics panel: shows real-time RAGAS-style scores (faithfulness estimate, # sources used)
- [ ] Ingestion page: drag-and-drop file upload with live progress
- [ ] Runs at `http://localhost:8501`

### Implementation instructions

**`ui/app.py`**: Main Streamlit app. Use `st.set_page_config(layout="wide")`. Three pages via `st.navigation`: "Chat", "Ingest Documents", "Evaluation Results". Use `st.session_state` for conversation history.

**Chat page**: Two columns — left (70%) for the chat interface, right (30%) for the retrieval inspector. Chat interface: `st.chat_input`, `st.chat_message` for each turn. On query submission: call the LexRAG API, display the answer with inline citation links `[1]`, `[2]`. Right panel: for the most recent query, display the `RetrievalResult` objects as expandable cards showing chunk text, score, retrieval rank, reranker rank. Add a "Show Entity Graph" button that renders a pyvis graph of related entities.

**Evaluation page**: Load all `results/ragas_*.json` files. Show a `st.dataframe` with all evaluation runs and their metric scores. Show a line chart (using `st.line_chart`) of each metric over time. Add a "Run New Evaluation" button that triggers the evaluation pipeline.

---

## GLOBAL RULES (apply to every phase)

**Error handling**: Every external call (Ollama, Qdrant, Neo4j, Groq, Redis) must be wrapped in try/except. Catch specific exceptions (not bare `except:`). Always log the full exception with `log.exception("what failed", extra_context=...)`. Never let an exception propagate silently.

**Type annotations**: Every function must have full type annotations. Run `mypy lexrag/ --strict` — it should pass with zero errors.

**Docstrings**: Every class and every public method must have a Google-style docstring explaining: what it does, its arguments, what it returns, and what exceptions it can raise.

**No magic numbers**: Every threshold, batch size, timeout value, and model parameter must be either in `config.py` (if it should be environment-configurable) or defined as a named constant at the top of the file (if it's a fixed implementation detail). Never hardcode `0.75` or `512` inline.

**Testing**: Every new class must have at minimum one unit test. Every integration point (Qdrant, Neo4j, external API) must have one integration test that uses a real local service (not mocks). Mock external APIs (Groq) in unit tests.

**Git discipline**: After each phase passes its test command, commit with a message following Conventional Commits: `feat(ingestion): implement hierarchical PDF parser with OCR fallback`. Never commit `.env` or model weights.

---

## EVALUATION TARGETS (what success looks like)

After completing all phases, your system should achieve:

| Metric | Target | How to measure |
|---|---|---|
| Context Precision | ≥ 0.85 | RAGAS evaluation |
| Context Recall | ≥ 0.78 | RAGAS evaluation |
| Faithfulness | ≥ 0.87 | RAGAS evaluation |
| Answer Relevancy | ≥ 0.82 | RAGAS evaluation |
| Query P95 latency | < 4 seconds | Prometheus histogram |
| Ingestion throughput | ≥ 10 pages/second | Manual timing |
| Hallucination rate | < 0.13 | 1 - Faithfulness |
| Re-ranker improvement | ≥ 15% precision gain | Ablation comparison |

---

## WHAT TO HIGHLIGHT IN YOUR PORTFOLIO

When presenting this project, emphasize these decisions as evidence of enterprise-level thinking:

1. **Hierarchical chunking over flat chunking** — explain the small-to-big retrieval pattern and why it beats fixed-size chunks for long documents

2. **Hybrid BM25 + dense retrieval with RRF** — show the ablation results proving it beats vector-only search for legal text with specific citations and clause numbers

3. **Cross-encoder re-ranking** — explain why bi-encoder similarity (used in retrieval) is fast but imprecise, and why cross-encoders (re-ranking) are slower but dramatically more accurate

4. **Graph RAG for multi-hop reasoning** — demonstrate a query that requires traversing entity relationships (e.g. "Who are the parties to the contract that governs the IBM-Accenture joint venture?") and show how vector-only RAG fails it while graph traversal succeeds

5. **RAGAS evaluation with ablation** — this is the most unusual thing in a portfolio. Most candidates show RAG systems. Almost none show *measured, improving* RAG systems. The trend chart of metric improvement is your strongest differentiator

6. **Full observability** — show the LangSmith trace for a complex query. Interviewers at enterprise companies know what production tracing looks like and are impressed when a candidate has it

---

*End of prompt file. Build phase by phase, test before proceeding, and treat each acceptance criteria checkbox as a hard requirement, not a suggestion.*
