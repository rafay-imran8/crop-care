# Phase 1: Model Training

This phase focuses on training three different transformer-based models for agricultural text processing tasks:
1. **Text Classifier** - Classifies agricultural text into categories
2. **Language Model (LM)** - Trains a decoder-based language model for text generation
3. **Sequence-to-Sequence (Seq2Seq)** - Trains an encoder-decoder model for converting agricultural paragraphs to checklists

## Project Structure

```
phase1/
├── data/
│   ├── checklist_pairs.py      # Sample (paragraph, checklist) pairs for seq2seq training
│   ├── classifier_data.py       # Sample labeled data for classifier training
│   └── __pycache__/
├── models/
│   ├── attention.py             # Multi-head attention mechanism
│   ├── decoder.py               # Decoder model (TinyGPT)
│   ├── encoder.py               # Encoder model for classification
│   ├── encoder_decoder.py       # Combined encoder-decoder for seq2seq
│   └── __pycache__/
├── train/
│   ├── train_classifier.py      # Train text classifier
│   ├── train_lm.py              # Train language model
│   ├── train_seq2seq.py         # Train sequence-to-sequence model
│   └── __pycache__/
├── utils/
│   ├── plotting.py              # Loss visualization utilities
│   ├── positional_encoding.py   # Positional encoding for transformers
│   ├── tokenizer.py             # Text tokenization utilities
│   └── __pycache__/
└── README.md
```

## Prerequisites

Install required dependencies:
```bash
pip install torch numpy scikit-learn
```

Or use the main requirements:
```bash
pip install -r ../requirements.txt
```

## Models Overview

### 1. Text Classifier (`train_classifier.py`)
**Purpose**: Classifies agricultural text into predefined categories

**Architecture**:
- Encoder-based transformer model
- Multi-head attention with 4 heads
- 2 transformer layers
- Model size: 128 dimensions

**Data**: Loaded from `data/classifier_data.py` - contains labeled samples with their categories

**How to Run**:
```bash
cd phase1/train
python train_classifier.py
```

**Output**:
- Trained model (saved to disk)
- Training loss plot
- Classification accuracy metrics

**Hyperparameters**:
- Max sequence length: 128 tokens
- Epochs: 30
- Learning rate: 0.001
- Device: GPU (if available) or CPU

---

### 2. Language Model (`train_lm.py`)
**Purpose**: Generates agricultural text by learning language patterns from a corpus

**Architecture**:
- Decoder-only transformer (TinyGPT)
- GPT-style autoregressive language model
- Causal self-attention (can only attend to previous tokens)

**Data**: Corpus of agricultural knowledge hardcoded in the script about:
- Wheat cultivation and disease management
- Maize growth and pest control
- Irrigation and fertilizer management
- Integrated pest management

**How to Run**:
```bash
cd phase1/train
python train_lm.py
```

**Output**:
- Trained language model
- Training loss plot
- Generated text samples

**Hyperparameters**:
- Vocabulary size: Varies based on corpus
- Embedding dimension: 64
- Model layers: 2
- Epochs: 50
- Learning rate: 0.001

**Example Generated Text**:
```
"wheat crop yield improves with balanced nitrogen application during..."
```

---

### 3. Sequence-to-Sequence Model (`train_seq2seq.py`)
**Purpose**: Converts agricultural paragraphs to structured checklists

**Architecture**:
- Encoder: Transforms input paragraph to context vectors
- Decoder: Generates output checklist tokens from context
- Encoder-Decoder Transformer with cross-attention

**Data**: Loaded from `data/checklist_pairs.py` - contains:
- Input: Agricultural paragraphs (variable length)
- Output: Structured checklists (variable length)

**Example**:
```
Input: "Wheat requires moderate irrigation and fertile soil..."
Output: "water soil monitor disease resistant varieties..."
```

**How to Run**:
```bash
cd phase1/train
python train_seq2seq.py
```

**Output**:
- Trained encoder-decoder model
- Training loss plot
- Checklist generation samples

**Hyperparameters**:
- Encoder max length: 12 tokens (for paragraphs)
- Decoder max length: 20 tokens (for checklists)
- Embedding dimension: 128
- Attention heads: 4
- Epochs: 30
- Learning rate: 0.001

---

## Training Tips

### GPU Acceleration
All models support GPU acceleration via PyTorch:
```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

The models will automatically use GPU if available, otherwise fall back to CPU.

### Monitoring Training
Each training script generates a loss plot (`loss_plot.png`) in the `train/` directory to visualize:
- Training progress
- Convergence behavior
- Potential overfitting

### Adjusting Hyperparameters

Edit the hyperparameter section in each training script:
```python
EPOCHS = 30          # Number of training epochs
LR = 1e-3           # Learning rate
MAX_LEN = 128       # Maximum sequence length
```

Lower learning rates (0.0001) = slower but more stable training
Higher learning rates (0.01) = faster but potentially unstable training

### Custom Data

To train on custom data:

**For Classifier**: Edit `data/classifier_data.py`
```python
samples = [
    ("agricultural text", category_index),
    ("more text", another_category),
]
```

**For Language Model**: Edit the `corpus` variable in `train_lm.py`

**For Seq2Seq**: Edit `data/checklist_pairs.py`
```python
samples = [
    ("paragraph text", "checklist output"),
    ("more paragraphs", "more checklists"),
]
```

---

## Model Utilities

### Tokenizer (`utils/tokenizer.py`)
- Simple word-based tokenizer
- Builds vocabulary from training data
- Handles padding and special tokens

### Positional Encoding (`utils/positional_encoding.py`)
- Adds position information to embeddings
- Enables transformer to understand word order
- Uses sine/cosine functions (standard transformer approach)

### Attention Mechanism (`models/attention.py`)
- Multi-head self-attention
- Allows model to attend to different parts of sequence
- Critical for transformer performance

### Plotting (`utils/plotting.py`)
- Visualizes training loss over epochs
- Helps identify training issues
- Saves plots for reference

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `MAX_LEN`, batch size, or use CPU |
| Very slow training | Use GPU instead of CPU; reduce model size |
| Loss not decreasing | Increase learning rate or check data format |
| File not found errors | Run scripts from `phase1/train/` directory |

---

## Next Steps

After training Phase 1 models:
1. Evaluate models on test set
2. Fine-tune hyperparameters based on performance
3. Save best model checkpoints
4. Move to Phase 2 (RAG system) that may use trained embeddings/representations

---

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Sentence Transformers](https://www.sbert.net/)
