<<<<<<< HEAD
=======

>>>>>>> d55ac52 (Add production deployment files and complete pipeline configuration)
# RAG Pipeline Exercise

This exercise involves implementing a Retrieval-Augmented Generation (RAG) pipeline to answer questions about Equal Experts case studies. The pipeline combines document chunking, embeddings, vector search, and LLM-based question answering.

## The Challenge

As a starting point, you need to implement:

1. The `chunk_text` method in `SimpleChunker` class (`src/chunking/chunking_strategies.py`)
   - Input: A text string
   - Output: A list of text chunks
   - Consider chunk size and overlap parameters

2. The `add_documents` method in `RAGPipeline` class (`src/rag/pipeline.py`)
   - Load the data from `case_studies.json`
   - Input: A list of documents
   - Process: Chunk documents → Generate embeddings → Store in vector store
   - Replace the current fixed text implementation

This should be enough to get the basic pipeline working.

As a next step, you should implement evaluation metrics to evaluate the performance of the pipeline, you can use external tools if you would like.

As an end goal, you should implement the `AdvancedChunker` class and the `chunk_text` method in the `AdvancedChunker` class.

And evaluate the performance of the pipeline again. Also consider any prompt improvements you might have.

### Tips

✅ Keep the code simple

✅ Include unit tests

### Questions
- How do you think the pipeline could be improved?
- What are the trade-offs between the chunking strategies?
- What do you do to take the pipeline into production?

## Prerequisites

1. Git
2. Python 3.12
3. Poetry (dependency management)
4. Ollama (local LLM service)

## Setting up the environment

### Installing dependencies

```bash
poetry install
```

### Activating the shell

```bash
poetry shell
```

### Running the tests

```bash
pytest
```

### Running the pipeline

#### Installing Ollama

Check the [Ollama website](https://ollama.com) for your operating system.
##### Downloading models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```
#### Running the pipeline locally
Make sure you have the environment activated and ollama running with the models downloaded.

```bash
python -m src.main run
```

#### Running with Docker
Build and run the container locally:

```bash
docker build -t rag-pipeline .
docker run --rm rag-pipeline
```

#### Deploying with Render
1. Create a new Web Service on Render.
2. Connect your GitHub repository.
3. Set the build command to:

```bash
./scripts/render_build.sh
```

4. Set the start command to:

```bash
python -m src.main run
```

5. Add any required environment variables, such as:
   - `OLLAMA_BASE_URL`
   - `CASE_STUDIES_PATH`
   - `COLLECTION_NAME`

#### Publish Docker image to Docker Hub
1. Create a Docker Hub repository.
2. Add the secrets `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` to GitHub Actions.
3. Push to `main`, `master`, or `solution` to trigger the publish workflow.

### Developing Tooling

#### Types
```bash
poetry run mypy src/
``` 

#### ruff
```bash
poetry run ruff check --fix .
```
#### isort
```bash
poetry run isort src/ tests/
```
#### black
```bash
poetry run black src/ tests/
```

