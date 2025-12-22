import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# -------------------------
# Add phase1 directory to Python path
# -------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PHASE1_DIR)

# -------------------------
# Imports from your project
# -------------------------
from data.classifier_data import samples, label_map
from utils.tokenizer import SimpleTokenizer
from models.encoder import EncoderClassifier
from utils.plotting import plot_loss

# -------------------------
# Hyperparameters
# -------------------------
MAX_LEN = 128
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Prepare data
# -------------------------
texts = [s[0] for s in samples]
labels = torch.tensor([s[1] for s in samples]).to(DEVICE)

tokenizer = SimpleTokenizer(texts)
encoded = [tokenizer.encode(t, MAX_LEN) for t in texts]
inputs = torch.tensor(encoded).to(DEVICE)

# -------------------------
# Model, loss, optimizer
# -------------------------
model = EncoderClassifier(
    vocab_size=len(tokenizer.stoi),
    d_model=128,
    layers=2,
    heads=4
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# -------------------------
# Training loop
# -------------------------
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(inputs)               # (B, C)
    loss = criterion(outputs, labels)     # compute loss
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# -------------------------
# Evaluation on training data
# -------------------------
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(inputs), dim=1)

print("\nPredictions on training data:")
for i, text in enumerate(texts):
    print(f"{text} → {label_map[preds[i].item()]}")

# -------------------------
# Test on 10 unseen examples
# -------------------------
unseen_texts = [
    "The leaves are turning yellow and falling off",
    "There is a lot of rust on the wheat leaves",
    "The maize crop looks healthy and tall",
    "Spots are appearing on the rice plants",
    "The plant growth is slow due to lack of sunlight",
    "Wheat stems are bending and weak",
    "Rice paddies are flooded and waterlogged",
    "The maize leaves have holes and damage",
    "Yellowing of leaves in the rice field observed",
    "The wheat field is dry and showing signs of stress"
]

print("\nPredictions on unseen data:")
for text in unseen_texts:
    encoded_text = torch.tensor([tokenizer.encode(text, MAX_LEN)]).to(DEVICE)
    predicted_label = torch.argmax(model(encoded_text), dim=1)
    print(f"{text} → {label_map[predicted_label.item()]}")

# -------------------------
# Plot training loss
# -------------------------
plot_loss(losses)
