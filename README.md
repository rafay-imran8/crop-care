Yes — this root README is outdated in several places, especially **OpenRouter → Groq**, model names, chunking, file names, and the current Phase 1/Phase 2 structure. Here is the modified version while keeping your existing README structure and content style.

# Agricultural AI: Multi-Phase NLP & RAG System

A comprehensive project implementing transformer-based models for agricultural text processing and a Retrieval-Augmented Generation (RAG) system for answering agricultural questions.

## Project Overview

This project is divided into two main phases:

### Phase 1: Transformer Model Training

Implements three transformer-based architectures **from scratch for educational and practical understanding**:

* **Text Classifier** — Classifies agricultural text into predefined categories
* **Language Model** — Decoder-only transformer for agricultural text generation
* **Sequence-to-Sequence Model** — Encoder-decoder transformer for converting agricultural paragraphs into checklists

### Phase 2: RAG System

Implements a complete Retrieval-Augmented Generation pipeline:

1. Extracts text and metadata from agricultural PDF documents
2. Splits documents into smaller chunks
3. Generates vector embeddings for each chunk
4. Stores embeddings and metadata locally
5. Retrieves relevant chunks using semantic similarity
6. Routes retrieval based on crop when possible
7. Generates grounded answers using a Groq-hosted LLM
8. Evaluates retrieval performance using predefined queries

## Project Structure

```text
.
├── README.md
├── requirements.txt
│
├── phase1/
│   ├── README.md
│   ├── data/
│   │   ├── checklist_pairs.py
│   │   ├── classifier_data.py
│   │   └── corpus.py
│   │
│   ├── models/
│   │   ├── attention.py
│   │   ├── decoder.py
│   │   ├── encoder.py
│   │   └── encoder_decoder.py
│   │
│   ├── train/
│   │   ├── train_classifier.py
│   │   ├── train_lm.py
│   │   └── train_seq2seq.py
│   │
│   └── utils/
│       ├── decoder_tokenizer.py
│       ├── plotting.py
│       ├── positional_encoding.py
│       ├── seq2seqtokenizer.py
│       └── tokenizer.py
│
└── phase2/
    ├── README.md
    ├── run_rag.py
    │
    ├── data/
    │   ├── raw/
    │   │   └── *.pdf
    │   └── processed/
    │       ├── documents.json
    │       └── chunks.json
    │
    ├── embeddings/
    │   └── embed_store.py
    │
    ├── ingestion/
    │   ├── load_docs.py
    │   └── chunk_docs.py
    │
    ├── retrieval/
    │   └── retriever.py
    │
    ├── generation/
    │   ├── generate.py
    │   └── prompt.py
    │
    └── evaluation/
        ├── queries.json
        └── evaluate_retrieval.py
```

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Phase 2 also requires environment-variable support:

```bash
pip install python-dotenv
```

### 3. Configure Groq API Key

Phase 2 uses the **Groq API** for LLM-based answer generation.

#### Option A: Environment Variable

**Windows PowerShell:**

```powershell
$env:GROQ_API_KEY = "your-api-key-here"
```

**Linux/macOS:**

```bash
export GROQ_API_KEY="your-api-key-here"
```

#### Option B: `.env` File

Create a `.env` file in the `phase2/` directory:

```text
GROQ_API_KEY=your-api-key-here
```

Do not commit the `.env` file to Git.

## Quick Start

### Phase 1: Training Models

Run the training scripts from the `phase1` directory.

#### Train Text Classifier

```bash
cd phase1
python train/train_classifier.py
```

Trains an encoder-based transformer for agricultural text classification.

#### Train Language Model

```bash
cd phase1
python train/train_lm.py
```

Trains a decoder-only transformer language model and generates agricultural text.

#### Train Sequence-to-Sequence Model

```bash
cd phase1
python train/train_seq2seq.py
```

Trains an encoder-decoder transformer to convert agricultural paragraphs into structured checklists.

### Phase 2: Running the RAG System

#### 1. Add Agricultural Documents

Place PDF documents in:

```text
phase2/data/raw/
```

The ingestion pipeline associates documents with metadata such as:

* Crop
* Publication year
* Source filename
* Page number

#### 2. Extract Documents

From the `phase2` directory:

```bash
python ingestion/load_docs.py
```

This extracts text from the PDFs and creates:

```text
phase2/data/processed/documents.json
```

#### 3. Create Chunks

```bash
python ingestion/chunk_docs.py
```

This creates:

```text
phase2/data/processed/chunks.json
```

Each chunk contains its text, unique chunk ID, and document metadata.

#### 4. Generate Embeddings

```bash
python embeddings/embed_store.py
```

The system uses the `all-MiniLM-L6-v2` Sentence Transformer model to generate embeddings and stores them in:

```text
phase2/embeddings/embed_store.json
```

#### 5. Run the RAG Application

```bash
python run_rag.py
```

The application will:

* Accept agricultural questions
* Generate an embedding for the question
* Route the query by crop when possible
* Retrieve relevant document chunks
* Display retrieved sources and similarity scores
* Send the retrieved context to the Groq LLM
* Generate a grounded answer
* Provide source citations in the response

#### 6. Evaluate Retrieval

```bash
python evaluation/evaluate_retrieval.py
```

The evaluation uses predefined queries from:

```text
phase2/evaluation/queries.json
```

and produces retrieval metrics in:

```text
phase2/evaluation/retrieval_metrics.json
```

## Key Technologies

### Deep Learning & NLP

* **PyTorch** — Neural network framework
* **Custom Transformers** — Encoder, decoder, and encoder-decoder architectures implemented for educational purposes
* **Sentence Transformers** — Semantic text embeddings
* **Custom Tokenizers** — Tokenization utilities used by Phase 1 models

### RAG Components

* **PDF Extraction** — PyMuPDF
* **Document Chunking** — Character-based overlapping chunks
* **Vector Embeddings** — `all-MiniLM-L6-v2`
* **Semantic Retrieval** — Cosine similarity
* **Query Routing** — Crop-based retrieval filtering
* **LLM Generation** — Groq API
* **Prompt Engineering** — Grounded context-based prompting
* **Local Storage** — JSON-based document and embedding storage

### Supporting Tools

* **NumPy** — Vector and similarity calculations
* **JSON** — Local persistence of documents, chunks, and embeddings
* **Scikit-learn** — Machine learning utilities
* **PyMuPDF** — PDF text extraction

## File Descriptions

### Phase 1 Key Files

| File                           | Purpose                                         |
| ------------------------------ | ----------------------------------------------- |
| `train/train_classifier.py`    | Train the transformer-based text classifier     |
| `train/train_lm.py`            | Train the decoder-only language model           |
| `train/train_seq2seq.py`       | Train the encoder-decoder transformer           |
| `models/attention.py`          | Multi-head attention implementation             |
| `models/encoder.py`            | Transformer encoder architecture                |
| `models/decoder.py`            | Decoder-only transformer architecture           |
| `models/encoder_decoder.py`    | Encoder-decoder transformer architecture        |
| `utils/tokenizer.py`           | Tokenization utilities for encoder-based models |
| `utils/decoder_tokenizer.py`   | Tokenizer for the decoder language model        |
| `utils/seq2seqtokenizer.py`    | Tokenizer for the Seq2Seq model                 |
| `utils/positional_encoding.py` | Positional encoding implementation              |
| `utils/plotting.py`            | Training loss visualization                     |

### Phase 2 Key Files

| File                               | Purpose                                                |
| ---------------------------------- | ------------------------------------------------------ |
| `run_rag.py`                       | Main interactive RAG application                       |
| `ingestion/load_docs.py`           | Extract text and metadata from PDFs                    |
| `ingestion/chunk_docs.py`          | Split extracted documents into overlapping chunks      |
| `embeddings/embed_store.py`        | Generate and store vector embeddings                   |
| `retrieval/retriever.py`           | Filter, route, and retrieve relevant chunks            |
| `generation/generate.py`           | Generate answers using the Groq LLM                    |
| `generation/prompt.py`             | Build grounded prompts from retrieved context          |
| `evaluation/evaluate_retrieval.py` | Evaluate retrieval performance                         |
| `evaluation/queries.json`          | Ground-truth evaluation queries and relevant documents |

## Configuration & Hyperparameters

### Phase 1

The exact configuration is defined directly in the individual training scripts.

#### Text Classifier

* Maximum sequence length: 128 tokens
* Hidden dimension: 128
* Attention heads: 4
* Transformer layers: 2
* Epochs: 30
* Learning rate: 0.001

#### Language Model

* Maximum sequence length: 128 tokens
* Hidden dimension: 128
* Attention heads: 4
* Transformer layers: 2
* Epochs: 50
* Learning rate: 0.001

#### Seq2Seq Model

* Encoder maximum sequence length: 12 tokens
* Decoder maximum sequence length: determined from training data
* Hidden dimension: 128
* Attention heads: 4
* Encoder layers: 2
* Decoder layers: 2
* Epochs: 30
* Learning rate: 0.001

### Phase 2

#### Document Chunking

* Chunk size: 500 characters
* Chunk overlap: 50 characters

#### Embeddings

* Model: `all-MiniLM-L6-v2`
* Embeddings generated locally

#### Retrieval

* Top-k: 3 chunks
* Similarity metric: Cosine similarity
* Similarity threshold: 0.35
* Crop-based query routing when a supported crop is explicitly mentioned

#### Generation

* Provider: Groq
* Model: Configured in `generation/generate.py`
* Temperature: 0.2
* Maximum output tokens: 500

## Example Usage

### Phase 2: Asking Questions

After running:

```bash
python run_rag.py
```

enter a question such as:

```text
Enter your query: How can wheat rust diseases be managed?
```

The system:

```text
User Query
    ↓
Query Embedding
    ↓
Crop Detection
    ↓
Quality Filtering
    ↓
Semantic Retrieval
    ↓
Top-k Relevant Chunks
    ↓
Context + Query
    ↓
Groq LLM
    ↓
Grounded Answer
```

The application also displays the retrieved sources, page numbers, and similarity scores.

## Troubleshooting

| Issue                       | Solution                                                                                                  |
| --------------------------- | --------------------------------------------------------------------------------------------------------- |
| `ModuleNotFoundError`       | Run scripts from the appropriate project root, e.g. `python train/train_seq2seq.py` from `phase1`         |
| `GROQ_API_KEY` error        | Set `GROQ_API_KEY` as an environment variable or configure it in `.env`                                   |
| Groq `model_not_found`      | Check the model configured in `phase2/generation/generate.py` against the currently available Groq models |
| No PDFs found               | Place supported PDF files in `phase2/data/raw/`                                                           |
| No relevant documents found | Check that ingestion, chunking, and embedding generation completed successfully                           |
| Low retrieval similarity    | Improve document quality, chunking strategy, or embedding/retrieval configuration                         |
| CUDA out of memory          | Reduce model size, batch size, or sequence length, or run on CPU                                          |
| `?` shown for page          | Regenerate `chunks.json` using the current ingestion/chunking pipeline                                    |

## Dependencies

The main dependencies are listed in `requirements.txt`.

Key dependencies include:

* `torch` — Deep learning framework
* `numpy` — Numerical computing
* `scikit-learn` — Machine learning utilities
* `PyMuPDF` — PDF processing
* `sentence-transformers` — Text embeddings
* `openai` — OpenAI-compatible client used to access Groq
* `python-dotenv` — Environment variable management

## Future Enhancements

* [ ] Improve document chunking using sentence/paragraph-aware boundaries
* [ ] Implement hybrid retrieval using semantic search + BM25
* [ ] Add reranking of retrieved chunks
* [ ] Improve retrieval evaluation with Recall@K, MRR, and NDCG
* [ ] Fine-tune embeddings for agricultural terminology
* [ ] Add support for multiple languages
* [ ] Add caching for frequently asked questions
* [ ] Deploy the RAG pipeline as a REST API
* [ ] Create a web-based dashboard
* [ ] Add support for tables and figures extracted from agricultural documents
* [ ] Add multimodal agricultural image understanding

## Notes

* Phase 1 transformer architectures are implemented from scratch for educational and practical learning purposes.
* Phase 1 models are trained on relatively small sample datasets and are not intended for production use.
* Phase 2 uses locally generated embeddings; document text and embeddings are stored locally.
* Phase 2 uses the Groq API only for final LLM-based answer generation.
* RAG answers are instructed to remain grounded in the retrieved agricultural context and provide source citations.
* The quality of retrieval depends heavily on the quality of document extraction, chunking, and embeddings.

**Last Updated:** August 2026
