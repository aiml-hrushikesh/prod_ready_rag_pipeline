# RAG Pipeline — Questions & Answers

## Q1: How do you think the pipeline could be improved?

### Retrieval
- **Hybrid search**: Combine vector (semantic) search with BM25 (keyword) search. This handles cases where exact terminology matters (e.g. product names, dates) that embeddings alone may miss.
- **Re-ranking**: After retrieving the top-k chunks, pass them through a cross-encoder re-ranker (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`) to reorder by true relevance before passing to the LLM.
- **Metadata filtering**: Attach metadata to chunks (source URL, client name, industry) and pre-filter by metadata before vector search to focus retrieval on relevant documents.

### Chunking
- **Semantic chunking**: Detect topic boundaries based on embedding cosine similarity shifts between sentences rather than using fixed sizes. This keeps semantically coherent ideas in a single chunk.
- **Sliding window with dynamic overlap**: Compute overlap based on sentence boundaries rather than a fixed character count to avoid splitting mid-thought.

### Generation
- **Prompt chaining**: For complex questions, decompose into sub-questions (RAG-Fusion / step-back prompting), retrieve for each, then synthesise a final answer.
- **Citation / grounding**: Instruct the LLM to cite which chunk(s) supported each claim, aiding traceability.

### Infrastructure
- **Caching**: Cache embeddings for frequently queried documents to avoid redundant embedding calls.
- **Streaming responses**: Stream LLM tokens back to the user for a more responsive experience.

---

## Q2: What are the trade-offs between the chunking strategies?

| | **SimpleChunker** (character-based sliding window) | **AdvancedChunker** (recursive separator-aware) |
|---|---|---|
| **Chunk boundaries** | Fixed character count — may cut mid-word or mid-sentence | Respects paragraphs → sentences → words → characters in that priority |
| **Semantic coherence** | Low — chunks can break in the middle of a thought | High — chunks tend to be complete ideas/sentences |
| **Embedding quality** | Lower — truncated sentences confuse the embedding model | Higher — coherent text produces better vector representations |
| **Retrieval accuracy** | Lower — fragmented context can mislead the LLM | Higher — the LLM receives meaningful, complete context |
| **Chunk count** | Predictable and uniform | Variable — depends on document structure |
| **Implementation complexity** | Very simple, O(n) | Moderate — recursive splitting with merge pass |
| **When to use** | Quick prototyping, highly structured / uniform text | Production systems, natural-language documents |

---

## Q3: What do you do to take the pipeline into production?

### 1. Infrastructure
- **Replace Milvus Lite (local `.db` file) with a managed Milvus cluster** (e.g. Zilliz Cloud) or another production vector DB (Qdrant, Weaviate, Pinecone) with replication and backups.
- **Containerise the application** using Docker and deploy behind a REST API (FastAPI) or message queue for async ingestion.

### 2. Observability & Monitoring
- **Structured logging** (e.g. JSON logs with correlation IDs) for every query, retrieved chunks, and LLM call.
- **Latency and error tracking** via Prometheus/Grafana or an APM tool.
- **RAGAS metrics in CI**: Run faithfulness and relevancy evaluations automatically on a golden test set on every PR to catch regression.

### 3. Model Serving
- **Ollama in production**: Run Ollama as a sidecar or dedicated GPU service behind a load balancer. Scale horizontally based on request volume.
- **Model versioning**: Pin specific model digests (e.g. `llama3.2:3b-instruct-q4_K_M`) to ensure reproducible outputs across deployments.

### 4. Data Pipeline
- **Document ingestion pipeline**: Replace the one-shot `add_documents` call with a streaming/batch pipeline (e.g. Airflow DAG or Kafka consumer) that handles new/updated documents incrementally.
- **Deduplication**: Track document hashes to avoid re-embedding unchanged content.

### 5. Testing & Safety
- **Unit + integration tests in CI** (pytest) covering chunkers, pipeline, and the vector store layer.
- **Guardrails**: Add input validation and output filters to block prompt injection attempts and prevent PII leakage.
- **Rate limiting** on the API layer to prevent abuse.

### 6. Evaluation in Production
- **Online evaluation**: Log a sample of real Q&A pairs and run RAGAS metrics asynchronously as a background job.
- **Human-in-the-loop feedback**: Collect thumbs-up/down signals from end users and feed them back into prompt iteration and fine-tuning cycles.
