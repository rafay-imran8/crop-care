import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add phase1 directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PHASE1_DIR)

from data.classifier_data import samples, label_map
from utils.tokenizer import SimpleTokenizer
from models.encoder import EncoderClassifier
from utils.plotting import plot_loss

# -------------------------
# Hyperparameters
# -------------------------
MAX_LEN = 32
BATCH_SIZE = 4
EPOCHS = 30
LR = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Prepare data
# -------------------------
texts = [s[0] for s in samples]
labels = torch.tensor([s[1] for s in samples]).to(DEVICE)

tokenizer = SimpleTokenizer(texts) #applied tokenizer
encoded = [tokenizer.encode(t, MAX_LEN) for t in texts]
inputs = torch.tensor(encoded).to(DEVICE)
# print(inputs)
# -------------------------
# Model
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
    # print(inputs.shape)
    optimizer.zero_grad()
    outputs = model(inputs)       # (B, C)
    loss = criterion(outputs, labels) #calculating loss
    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# -------------------------
# Evaluation (Sanity Check)
# -------------------------
model.eval()
with torch.no_grad():
    preds = torch.argmax(model(inputs), dim=1)

print("\nPredictions:")
for i, text in enumerate(texts):
    print(f"{text} → {label_map[preds[i].item()]}")

# -------------------------
# Plot loss curve
# -------------------------
# Example of testing it on unseen data
unseen_text = "The leaves are turning yellow and falling off"
predicted_label=torch.argmax(model(torch.tensor([tokenizer.encode(unseen_text, MAX_LEN)]).to(DEVICE)), dim=1)
print(f"\nUnseen Text Prediction:\n{unseen_text} → {label_map[predicted_label.item()]}")
plot_loss(losses)
