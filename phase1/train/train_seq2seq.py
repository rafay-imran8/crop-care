# train_seq2seq.py
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add phase1 directory to Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE1_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PHASE1_DIR)

from models.encoder_decoder import EncoderDecoderTransformer  # Seq2Seq model
from utils.plotting import plot_loss
from data.checklist_pairs import samples  # [(paragraph, checklist), ...]

# -------------------------
# Decoder Tokenizer (embedded)
# -------------------------
class Seq2SeqTokenizer:
    def __init__(self, texts):
        self.sos = "<sos>"
        self.eos = "<eos>"
        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())
        self.stoi = {self.sos:0, self.eos:1}
        self.stoi.update({w:i+2 for i,w in enumerate(sorted(vocab))})
        self.itos = {i:w for w,i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[self.sos]] + [self.stoi[w] for w in text.lower().split()] + [self.stoi[self.eos]]

    def decode(self, tokens):
        return " ".join([self.itos[t] for t in tokens if t > 1])

# -------------------------
# Encoder tokenizer (existing)
# -------------------------
from utils.tokenizer import SimpleTokenizer

MAX_LEN = 32

encoder_texts = [p[0] for p in samples]
decoder_texts = [p[1] for p in samples]

enc_tokenizer = SimpleTokenizer(encoder_texts)
dec_tokenizer = Seq2SeqTokenizer(decoder_texts)

# -------------------------
# Encode encoder inputs
# -------------------------
enc_inputs = torch.tensor([enc_tokenizer.encode(t, MAX_LEN) for t in encoder_texts])

# -------------------------
# Encode decoder inputs and targets with padding
# -------------------------
dec_encoded = [dec_tokenizer.encode(t) for t in decoder_texts]
dec_max_len = max(len(seq) for seq in dec_encoded)

# inputs: remove last token (<eos>), pad to max length
dec_inputs = [seq[:-1] + [0]*(dec_max_len - len(seq)) for seq in dec_encoded]
# targets: remove first token (<sos>), pad to max length
dec_targets = [seq[1:] + [0]*(dec_max_len - len(seq)) for seq in dec_encoded]

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

criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding
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
unseen_paragraphs = [
    "timely irrigation improves maize growth and prevents drought stress",
    "aphids and locusts damage wheat and maize crops during summer",
    "balanced nitrogen and phosphorus application enhances soil fertility"
]

print("\n--- Generating checklists for unseen paragraphs ---")
for para in unseen_paragraphs:
    # Tokenize encoder input (same tokenizer as training)
    enc_input = torch.tensor([enc_tokenizer.encode(para, MAX_LEN)], device=device)
    
    # Start with <sos> token for decoder
    generated = torch.tensor([[dec_tokenizer.stoi["<sos>"]]], device=device)
    
    # Generate tokens auto-regressively
    with torch.no_grad():
        for _ in range(25):  # max 25 tokens
            logits = model(enc_input, generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1)
            if next_token.item() == dec_tokenizer.stoi["<eos>"]:
                break
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
    
    # Decode generated tokens
    checklist = dec_tokenizer.decode(generated[0].tolist())
    print(f"\nParagraph: {para}")
    print(f"Generated checklist: {checklist}")