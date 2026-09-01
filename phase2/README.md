# Phase 2: RAG System (Retrieval-Augmented Generation)

This phase implements a complete RAG (Retrieval-Augmented Generation) pipeline for answering agricultural questions by retrieving relevant documents and generating answers using an LLM.

## Project Structure

```
phase2/
├── data/
│   ├── raw/                    # Store PDF documents here
│   └── processed/
│       ├── documents.json      # Extracted text from PDFs
│       └── chunks.json         # Chunked documents with metadata
├── embeddings/
│   ├── embed_store.json        # Vector embeddings for all chunks
│   └── embed_store.py          # Script to generate embeddings
├── evaluation/
│   ├── evaluate_retrieval.py   # Evaluate retrieval quality
│   ├── evaluation_metrics.md   # Evaluation results
│   ├── queries.json            # Test queries
│   └── retrieval_metrics.json  # Retrieval performance metrics
├── generation/
│   ├── generate.py             # LLM-based answer generation
│   ├── prompt.py               # Prompt engineering utilities
│   └── __pycache__/
├── ingestion/
│   ├── load_docs.py            # Load PDFs and extract text
│   ├── chunk_docs.py           # Split documents into chunks
│   └── __pycache__/
├── retrieval/
│   ├── retriever.py            # Retrieve relevant chunks
│   └── __pycache__/
├── run_rag.py                  # Main interactive RAG application
├── api.py                      # FastAPI web backend
├── app_service.py              # Upload and chat orchestration
├── frontend/                   # React browser client
└── README.md
```

## Prerequisites

### 1. Install Dependencies
```bash
pip install -r ../requirements.txt
```

Additionally, install:
```bash
pip install python-dotenv
```

### 2. Set Up OpenRouter API Key
The system uses OpenRouter's free LLM API for answer generation.

**Option A: Environment Variable (Recommended)**
```powershell
# Windows PowerShell
$env:OPENROUTER_API_KEY = "your-api-key-here"
```

```bash
# Linux/macOS
export OPENROUTER_API_KEY="your-api-key-here"
```

**Option B: .env File**
Create a `.env` file in the `phase2/` directory:
```
OPENROUTER_API_KEY=your-api-key-here
```

To get an API key:
1. Go to [OpenRouter.ai](https://openrouter.ai)
2. Sign up (free)
3. Get your API key from Settings → Keys
4. Use the key (starts with `sk-or-v1-`)

---

## RAG Pipeline Components

### Web Application

From the repository root, start the web application with:

```bash
uvicorn api:app --app-dir phase2 --reload
```

Open `http://localhost:8000`. Upload one or more PDFs with crop and publication year metadata. Each upload is extracted, chunked, embedded, and appended to the existing store; chat responses include retrieved source chunks and similarity scores.

### 1. Document Ingestion (`ingestion/load_docs.py`)

**Purpose**: Extract text from PDF documents

**How to Run**:
```bash
cd phase2/ingestion
python load_docs.py
```

**Steps**:
1. Place PDF files in `data/raw/` directory
2. Edit PDF metadata mapping in `load_docs.py`:
   ```python
   PDF_METADATA = {
       "your_file.pdf": {"crop": "wheat", "publication_year": 2020},
   }
   ```
3. Run the script
4. Output: `data/processed/documents.json`

**Output Format**:
```json
[
  {
    "id": 0,
    "text": "Full text extracted from PDF...",
    "metadata": {
      "source": "wheat_disease_rust_2011.pdf",
      "crop": "wheat",
      "publication_year": 2011
    }
  }
]
```

---

### 2. Document Chunking (`ingestion/chunk_docs.py`)

**Purpose**: Split long documents into smaller chunks for efficient retrieval

**How to Run**:
```bash
cd phase2/ingestion
python chunk_docs.py
```

**Configuration** (in script):
```python
CHUNK_SIZE = 500      # Characters per chunk
CHUNK_OVERLAP = 50    # Overlap between chunks
```

**Output**: `data/processed/chunks.json` with overlapping text chunks and metadata

**Why Overlapping Chunks?**
- Prevents splitting important concept across chunks
- Ensures relevant information is fully captured
- Improves retrieval quality

---

### 3. Embedding Generation (`embeddings/embed_store.py`)

**Purpose**: Convert text chunks to vector embeddings for semantic search

**How to Run**:
```bash
cd phase2
python embeddings/embed_store.py
```

**Model Used**: `all-MiniLM-L6-v2` (from Sentence Transformers)
- Small and fast (22MB)
- Good for semantic similarity
- 384-dimensional embeddings
- Works on CPU

**Output**: `embeddings/embed_store.json` containing:
```json
[
  {
    "id": 0,
    "text": "chunk text...",
    "embedding": [0.123, -0.456, ...],  // 384 dimensions
    "metadata": {...}
  }
]
```

**Time Estimate**: 
- 100 chunks: ~5 seconds
- 1000 chunks: ~30 seconds

---

### 4. Retrieval (`retrieval/retriever.py`)

**Purpose**: Find most relevant chunks for a user query

**Configuration** (in `run_rag.py`):
```python
SIMILARITY_THRESHOLD = 0.35  # Minimum similarity score (0-1)
TOP_K = 3                     # Number of chunks to retrieve
```

**How it Works**:
1. Encode user query using same embedding model
2. Compute cosine similarity with all chunk embeddings
3. Return top K chunks with highest similarity
4. Filter by minimum threshold (low-confidence fallback)

**Similarity Score Interpretation**:
- 0.8-1.0: Highly relevant
- 0.5-0.8: Relevant
- 0.3-0.5: Somewhat relevant
- <0.3: Not relevant

---

### 5. Answer Generation (`generation/generate.py`)

**Purpose**: Generate natural language answers using retrieved context

**LLM Configuration**:
- **Provider**: OpenRouter
- **Model**: `google/gemma-7b-it` (free, high-quality)
- **Temperature**: 0.2 (more deterministic, less creative)
- **Max tokens**: 500 (answer length limit)

**Other Free Models to Try**:
- `meta-llama/llama-3.3-70b-instruct` (best quality)
- `qwen/qwen-7b-chat`
- `huggingface/zephyr-7b-beta`

**How to Switch Model**:
Edit `generation/generate.py`:
```python
model="google/gemma-7b-it",  # Change this
```

---

### 6. Prompt Engineering (`generation/prompt.py`)

**Purpose**: Format retrieval results and user query into effective LLM prompt

**Typical Prompt Structure**:
```
Context from retrieved documents:
[chunk 1]
[chunk 2]
[chunk 3]

Question: [user query]

Answer:
```

The prompt provides:
- Retrieved context to ground the answer
- Clear question being asked
- Format guidance for the LLM

---

## Running the RAG System

### Interactive Mode (Main Application)

```bash
cd phase2
python run_rag.py
```

**Features**:
- Ask up to 10 agricultural questions
- Type 'exit' to quit early
- Shows retrieval similarity scores
- Displays generated answers with source context
- Handles errors gracefully

**Example Usage**:
```
RAG System - Ask Your Agricultural Questions
============================================================
Enter up to 10 questions (type 'exit' to quit)

Question 1/10: What are the symptoms of leaf rust in wheat?
Processing: What are the symptoms of leaf rust in wheat?
Top similarity score: 0.823

--- RAG ANSWER ---
Leaf rust in wheat appears as small, elongated rust-colored pustules on the 
upper surface of leaves. The disease develops in cool, moist conditions...

------------------------------------------------------------

Question 2/10: How to manage maize pests?
...
```

---

## Complete Workflow

### First-Time Setup

1. **Prepare documents**:
   ```bash
   # Add PDF files to data/raw/
   ```

2. **Ingest documents**:
   ```bash
   cd phase2/ingestion
   python load_docs.py
   ```

3. **Chunk documents**:
   ```bash
   python chunk_docs.py
   ```

4. **Generate embeddings**:
   ```bash
   cd phase2
   python embeddings/embed_store.py
   ```

5. **Set API key**:
   ```powershell
   $env:OPENROUTER_API_KEY = "your-key"
   ```

6. **Run RAG system**:
   ```bash
   python run_rag.py
   ```

### Subsequent Runs

Once embeddings are generated:
```bash
cd phase2
python run_rag.py
```

---

## Configuration & Tuning

### Retrieve More/Fewer Results

Edit `run_rag.py`:
```python
TOP_K = 5  # Get 5 chunks instead of 3
```

More chunks = better context but slower, longer answers

### Adjust Similarity Threshold

```python
SIMILARITY_THRESHOLD = 0.5  # Higher = stricter relevance requirement
```

Higher threshold filters out low-confidence results:
- Too high: May reject valid answers
- Too low: May include irrelevant context

### Change Embedding Model

Edit `run_rag.py`:
```python
model = SentenceTransformer("all-mpnet-base-v2")  # Better but slower
```

Alternative models:
- `all-mpnet-base-v2`: Higher quality, larger
- `all-MiniLM-L6-v2`: Current (balanced)
- `paraphrase-MiniLM-L6-v2`: Faster, lower quality

### Adjust Answer Generation

Edit `generation/generate.py`:
```python
temperature=0.5,      # 0=deterministic, 1=creative
max_tokens=1000,      # Longer answers (uses more tokens)
```

---

## Evaluation

### Run Evaluation

```bash
cd phase2/evaluation
python evaluate_retrieval.py
```

Metrics computed:
- **Precision**: How many retrieved chunks are relevant
- **Recall**: How many relevant chunks were retrieved
- **MRR (Mean Reciprocal Rank)**: Position of first relevant result
- **nDCG (Normalized Discounted Cumulative Gain)**: Quality of ranking

### View Results

Check `evaluation/retrieval_metrics.json`:
```json
{
  "avg_precision": 0.78,
  "avg_recall": 0.82,
  "avg_mrr": 0.91,
  "avg_ndcg": 0.85
}
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "No relevant documents found" | Add more documents; lower `SIMILARITY_THRESHOLD` |
| API key errors | Set `OPENROUTER_API_KEY` environment variable |
| Slow response | Reduce `TOP_K`; use faster embedding model |
| Poor answer quality | Add more relevant documents; adjust prompt |
| PDF extraction errors | Ensure PDFs are text-based (not scanned images) |
| Memory issues | Process documents in smaller batches |

---

## API Costs

- **OpenRouter Gemma 7B**: Completely FREE
- **Embedding generation**: One-time cost (local, no API calls)
- **Retrieval**: No cost (local, no API calls)

The system is designed to be **completely free** - all computational costs are one-time, and API calls use free tier models.

---

## Advanced Usage

### Custom Prompt Templates

Edit `generation/prompt.py` to customize answer format:
```python
def build_prompt(context_blocks, question):
    context = "\n\n".join([f"Source {i+1}:\n{c['text']}" 
                           for i, c in enumerate(context_blocks)])
    
    return f"""You are an agricultural expert. Answer the question using ONLY 
    the provided context. If not found in context, say "Not available".
    
    Context:
    {context}
    
    Question: {question}
    
    Answer:"""
```

### Batch Processing

Modify `run_rag.py` to process queries from a file:
```python
with open("queries.txt") as f:
    queries = f.readlines()

for query in queries:
    # Run RAG pipeline
```

### Save Answers

Add logging to `run_rag.py`:
```python
with open("answers.json", "a") as f:
    json.dump({
        "question": query_text,
        "answer": answer,
        "score": top_score
    }, f)
    f.write("\n")
```

---

## Performance Tips

1. **Batch embedding generation**: Use GPU if available
2. **Reduce chunk size**: Faster retrieval, more chunks
3. **Use smaller embedding model**: Faster inference
4. **Cache embeddings**: Load once, reuse for multiple queries
5. **Optimize prompt**: Shorter prompts = faster generation

---

## References

- [RAG Overview](https://arxiv.org/abs/2005.11401)
- [Sentence Transformers](https://www.sbert.net/)
- [OpenRouter Documentation](https://openrouter.ai/docs)
- [Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
