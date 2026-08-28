import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    )
)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from data.corpus import corpus
from models.decoder import TinyGPT
from utils.decoder_tokenizer import DecoderTokenizer
from utils.plotting import plot_loss


# -------------------------
# Configuration
# -------------------------
MAX_LEN = 128
BATCH_SIZE = 32

D_MODEL = 128
LAYERS = 2
HEADS = 4

LEARNING_RATE = 1e-3
EPOCHS = 10
GENERATION_LENGTH = 60


# -------------------------
# Device
# -------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")


# -------------------------
# Tokenizer
# -------------------------
tokenizer = DecoderTokenizer(corpus)

encoded = tokenizer.encode(corpus)

print(f"Vocabulary size: {len(tokenizer.stoi)}")
print(f"Encoded sequence length: {len(encoded)}")


# -------------------------
# Build Training Sequences
# -------------------------
inputs = []
targets = []

for i in range(len(encoded) - MAX_LEN):
    sequence = encoded[i:i + MAX_LEN]

    inputs.append(sequence[:-1])
    targets.append(sequence[1:])


inputs = torch.tensor(inputs, dtype=torch.long)
targets = torch.tensor(targets, dtype=torch.long)

print(f"Training samples: {len(inputs)}")
print(f"Input shape: {inputs.shape}")
print(f"Target shape: {targets.shape}")


# -------------------------
# Dataset & DataLoader
# -------------------------
dataset = TensorDataset(inputs, targets)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)


# -------------------------
# Model
# -------------------------
model = TinyGPT(
    vocab_size=len(tokenizer.stoi),
    d_model=D_MODEL,
    layers=LAYERS,
    heads=HEADS,
    max_len=MAX_LEN,
).to(device)


# -------------------------
# Loss & Optimizer
# -------------------------
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# -------------------------
# Training
# -------------------------
losses = []

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0.0

    for batch_inputs, batch_targets in dataloader:

        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(batch_inputs)

        # Reshape for CrossEntropyLoss
        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            batch_targets.reshape(-1),
        )

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(dataloader)

    losses.append(average_loss)

    if epoch % 5 == 0 or epoch == EPOCHS - 1:
        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f}"
        )


# -------------------------
# Plot Training Loss
# -------------------------
plot_loss(losses)


# -------------------------
# Text Generation
# -------------------------
model.eval()

start_text = "agriculture helps"

generated_tokens = tokenizer.encode(
    start_text,
)

generated = torch.tensor(
    [generated_tokens],
    dtype=torch.long,
    device=device,
)


for _ in range(GENERATION_LENGTH):

    # Keep only the most recent context
    context = generated[:, -MAX_LEN:]

    with torch.no_grad():
        logits = model(context)

        # Get prediction for the next token
        next_token = torch.argmax(
            logits[:, -1, :],
            dim=-1,
        )

    generated = torch.cat(
        [
            generated,
            next_token.unsqueeze(1),
        ],
        dim=1,
    )

    # Stop if EOS is generated
    if next_token.item() == tokenizer.stoi[tokenizer.eos]:
        break


# -------------------------
# Output
# -------------------------
print("\nGenerated text:")
print(
    tokenizer.decode(
        generated[0].tolist()
    )
)