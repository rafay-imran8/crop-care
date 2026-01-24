# Agricultural AI: Multi-Phase NLP & RAG System

A comprehensive project implementing transformer-based models for agricultural text processing and a Retrieval-Augmented Generation (RAG) system for answering agricultural questions.

## Project Overview

This project is divided into two main phases:

### Phase 1: Model Training
Trains three different transformer-based models for agricultural text understanding and generation:
- **Text Classifier** - Categorizes agricultural text
- **Language Model** - Generates agricultural text
- **Sequence-to-Sequence Model** - Converts agricultural paragraphs to checklists

### Phase 2: RAG System
Implements a complete RAG pipeline that:
1. Ingests and chunks agricultural documents
2. Embeds chunks into vector space
3. Retrieves relevant documents for user queries
4. Generates contextual answers using an LLM

## Project Structure

```
.
├── README.md                  # This file
├── requirements.txt           # Python dependencies
│
├── phase1/                    # Model Training
│   ├── README.md
│   ├── data/                  # Training data samples
│   ├── models/                # Transformer model implementations
│   ├── train/                 # Training scripts
│   └── utils/                 # Utilities (tokenizer, positional encoding, etc.)
│
└── phase2/                    # RAG System
    ├── README.md
    ├── run_rag.py             # Main interactive application
    ├── data/                  # Documents and chunks
    ├── embeddings/            # Vector embeddings
    ├── evaluation/            # Evaluation scripts and metrics
    ├── generation/            # LLM-based answer generation
    ├── ingestion/             # Document loading and chunking
    └── retrieval/             # Retrieval mechanisms
```

## Installation

### 1. Clone/Download the Repository

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

Additional dependencies for Phase 2:
```bash
pip install python-dotenv
```

### 3. Configure API Key (Phase 2 only)
The Phase 2 RAG system uses OpenRouter's free LLM API.

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
Create `.env` in the `phase2/` directory:
```
OPENROUTER_API_KEY=your-api-key-here
```

Get your free API key at [OpenRouter.ai](https://openrouter.ai)

## Quick Start

### Phase 1: Training Models

#### Train Text Classifier
```bash
cd phase1/train
python train_classifier.py
```
Outputs: Trained classifier model and performance metrics

#### Train Language Model
```bash
cd phase1/train
python train_lm.py
```
Outputs: Trained LM and generated text samples

#### Train Sequence-to-Sequence Model
```bash
cd phase1/train
python train_seq2seq.py
```
Outputs: Trained seq2seq model for paragraph-to-checklist conversion

### Phase 2: Running the RAG System

#### 1. Ingest Documents
```bash
cd phase2/ingestion
# First, place PDF files in ../data/raw/
python load_docs.py       # Extract text from PDFs
cd ../
python chunk_docs.py      # Split into chunks
```

#### 2. Generate Embeddings
```bash
cd phase2/embeddings
python embed_store.py
```

#### 3. Run Interactive RAG Application
```bash
cd phase2
python run_rag.py
```

The application will:
- Prompt you for agricultural questions
- Retrieve relevant document chunks
- Generate answers using the LLM
- Display sources and confidence scores

#### 4. Evaluate System Performance
```bash
cd phase2/evaluation
python evaluate_retrieval.py
```

## Key Technologies

### Deep Learning & NLP
- **PyTorch** - Neural network framework
- **Transformers** - Multi-head attention, encoder-decoder architectures
- **Tokenization** - Custom word-based tokenizer

### RAG Components
- **Vector Embeddings** - Text-to-embedding conversion
- **Semantic Search** - Find relevant chunks via similarity
- **LLM Integration** - OpenRouter API for answer generation
- **Prompt Engineering** - Context-aware prompts for generation

### Supporting Tools
- **PDF Processing** - Extract text from documents
- **JSON Storage** - Persist embeddings and metadata
- **Evaluation Metrics** - Precision, recall, MRR, NDCG

## File Descriptions

### Phase 1 Key Files

| File | Purpose |
|------|---------|
| `train/train_classifier.py` | Train text classifier with labeled data |
| `train/train_lm.py` | Train decoder-only language model |
| `train/train_seq2seq.py` | Train encoder-decoder for paragraph→checklist |
| `models/attention.py` | Multi-head attention implementation |
| `models/encoder_decoder.py` | Seq2seq architecture |
| `utils/tokenizer.py` | Text tokenization and encoding |
| `utils/positional_encoding.py` | Positional embeddings for transformers |

### Phase 2 Key Files

| File | Purpose |
|------|---------|
| `run_rag.py` | Interactive RAG application (main entry point) |
| `ingestion/load_docs.py` | Extract text from PDF documents |
| `ingestion/chunk_docs.py` | Split documents into chunks |
| `embeddings/embed_store.py` | Generate and store embeddings |
| `retrieval/retriever.py` | Retrieve chunks for queries |
| `generation/generate.py` | Generate answers with LLM |
| `generation/prompt.py` | Prompt engineering utilities |
| `evaluation/evaluate_retrieval.py` | Evaluate retrieval performance |

## Configuration & Hyperparameters

### Phase 1 Model Parameters

**Text Classifier**
- Max sequence length: 128 tokens
- Hidden dimensions: 128
- Attention heads: 4
- Layers: 2
- Epochs: 30
- Learning rate: 0.001

**Language Model**
- Max sequence length: 128 tokens
- Hidden dimensions: 256
- Attention heads: 4
- Layers: 2
- Epochs: 40
- Learning rate: 0.001

**Seq2Seq Model**
- Encoder max length: 12 tokens
- Decoder max length: 20 tokens
- Shared embedding dimensions: 128
- Attention heads: 4
- Epochs: 30

### Phase 2 RAG Parameters

**Document Chunking**
- Chunk size: Configurable (default ~500 tokens)
- Overlap: To preserve context

**Retrieval**
- Top-k results: 5 chunks
- Similarity metric: Cosine distance

**Generation**
- Model: Llama 2 / Mistral (via OpenRouter)
- Max tokens: 500
- Temperature: 0.7

## Example Usage

### Phase 2: Asking Questions

Once the RAG system is running:

```
Enter your agricultural question: What are the best practices for wheat cultivation?

[System retrieves relevant chunks from documents]
[LLM generates answer based on retrieved context]

Answer: Best practices for wheat cultivation include...
Sources: [Document 1, Section 2.1], [Document 3, Section 4.2]
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'torch'` | Install PyTorch: `pip install torch` |
| `API key errors` | Set `OPENROUTER_API_KEY` environment variable or create `.env` file |
| `No PDFs found in data/raw/` | Place PDF files in `phase2/data/raw/` directory |
| `CUDA out of memory` | Reduce batch size or use CPU by setting `DEVICE = 'cpu'` |
| `Low retrieval accuracy` | Increase chunk size or use more documents |

## Dependencies

Core dependencies listed in `requirements.txt`:
- `torch` - Deep learning framework
- `numpy` - Numerical computing
- `scikit-learn` - ML utilities
- `pypdf` - PDF processing
- `python-dotenv` - Environment variable management
- `requests` - HTTP client for API calls

## Future Enhancements

- [ ] Fine-tune embeddings on agricultural domain
- [ ] Add support for multiple languages
- [ ] Implement hybrid search (semantic + BM25)
- [ ] Add caching for frequently asked questions
- [ ] Deploy as REST API
- [ ] Create web UI dashboard
- [ ] Add multi-modal support (images, tables)

## Notes

- The project uses custom transformer implementations for educational purposes
- Phase 1 models are trained on sample data; scale up with more agricultural corpora for production use
- Phase 2 relies on free OpenRouter API; consider alternative LLMs for production deployments
- All embeddings are computed locally; no external embedding service required

## Contact & Support

For questions or issues, refer to the individual README files:
- [Phase 1 Details](phase1/README.md)
- [Phase 2 Details](phase2/README.md)

---

**Last Updated**: January 2026
