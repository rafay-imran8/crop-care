# train_seq2seq.py
import torch
import torch.nn as nn
import torch.optim as optim
import sys, os

# Add phase1 directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PHASE1_DIR)

from models.encoder_decoder import EncoderDecoderTransformer
from utils.plotting import plot_loss
from data.checklist_pairs import samples  # [(paragraph, checklist), ...]

# -------------------------
# Decoder Tokenizer with PAD, SOS, EOS
# -------------------------
class Seq2SeqTokenizer:
    def __init__(self, texts):
        self.pad = "<pad>"
        self.sos = "<sos>"
        self.eos = "<eos>"

        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())

        # Indices: 0=pad, 1=sos, 2=eos, 3+=words
        self.stoi = {self.pad: 0, self.sos: 1, self.eos: 2}
        self.stoi.update({w: i + 3 for i, w in enumerate(sorted(vocab))})
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[self.sos]] + [self.stoi[w] for w in text.lower().split()] + [self.stoi[self.eos]]

    def decode(self, tokens):
        return " ".join([self.itos[t] for t in tokens if t > 2])

# -------------------------
# Prepare Data
# -------------------------
encoder_texts = [p[0] for p in samples]
decoder_texts = [p[1] for p in samples]

from utils.tokenizer import SimpleTokenizer
MAX_LEN = 12 # max length for encoder inputs, could be increased if we have more data

enc_tokenizer = SimpleTokenizer(encoder_texts)
dec_tokenizer = Seq2SeqTokenizer(decoder_texts)

# Encode encoder inputs
enc_inputs = torch.tensor([enc_tokenizer.encode(t, MAX_LEN) for t in encoder_texts])

# Encode decoder inputs and targets
dec_encoded = [dec_tokenizer.encode(t) for t in decoder_texts]
dec_max_len = max(len(seq) for seq in dec_encoded)

# Decoder inputs: remove <eos>, pad
dec_inputs = [seq[:-1] + [dec_tokenizer.stoi["<pad>"]] * (dec_max_len - len(seq)) for seq in dec_encoded]

# Decoder targets: remove <sos>, pad
dec_targets = [seq[1:] + [dec_tokenizer.stoi["<pad>"]] * (dec_max_len - len(seq)) for seq in dec_encoded]

# Convert to tensors
dec_inputs = torch.tensor(dec_inputs)
dec_targets = torch.tensor(dec_targets)

# -------------------------
# Model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = EncoderDecoderTransformer(
    enc_vocab_size=len(enc_tokenizer.stoi),
    dec_vocab_size=len(dec_tokenizer.stoi),
    d_model=128,
    enc_layers=2,
    dec_layers=2,
    heads=4,
    max_len=MAX_LEN
).to(device)

enc_inputs, dec_inputs, dec_targets = enc_inputs.to(device), dec_inputs.to(device), dec_targets.to(device)
print(enc_inputs)
print(dec_inputs)
print(dec_targets)
# Loss ignores padding
criterion = nn.CrossEntropyLoss(ignore_index=dec_tokenizer.stoi["<pad>"])
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Training
# -------------------------
EPOCHS = 30
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(enc_inputs, dec_inputs)  # (B, T, dec_vocab)
    loss = criterion(logits.view(-1, logits.size(-1)), dec_targets.view(-1))
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

plot_loss(losses)

# -------------------------
# Generation (auto-regressive)
# -------------------------
model.eval()
prompt = torch.tensor([[dec_tokenizer.stoi["<sos>"]]], device=device)
generated = prompt

with torch.no_grad():
    for _ in range(25):  # max 25 tokens
        logits = model(enc_inputs[:1], generated)  # use first paragraph
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        if next_token.item() == dec_tokenizer.stoi["<eos>"]:
            break
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

print("\nGenerated checklist:")
print(dec_tokenizer.decode(generated[0].tolist()))

# -------------------------
# Generate for unseen paragraphs
# -------------------------
unseen_paragraphs = [
    "timely irrigation improves maize growth and prevents drought stress",
    "aphids and locusts damage wheat and maize crops during summer",
    "balanced nitrogen and phosphorus application enhances soil fertility"
]

print("\n--- Generating checklists for unseen paragraphs ---")
for para in unseen_paragraphs:
    enc_input = torch.tensor([enc_tokenizer.encode(para, MAX_LEN)], device=device)
    generated = torch.tensor([[dec_tokenizer.stoi["<sos>"]]], device=device)

    with torch.no_grad():
        for _ in range(25):
            logits = model(enc_input, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == dec_tokenizer.stoi["<eos>"]:
                break
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

    print(f"\nParagraph: {para}")
    print(f"Generated checklist: {dec_tokenizer.decode(generated[0].tolist())}")
