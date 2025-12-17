import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim

from models.decoder import TinyGPT
from utils.plotting import plot_loss


# -------------------------
# Decoder Tokenizer
# -------------------------
class DecoderTokenizer:
    def __init__(self, text):
        self.sos = "<sos>"
        self.eos = "<eos>"

        words = text.split()
        vocab = sorted(set(words))

        self.stoi = {self.sos: 0, self.eos: 1}
        self.stoi.update({w: i + 2 for i, w in enumerate(vocab)})
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[self.sos]] + \
               [self.stoi[w] for w in text.split()] + \
               [self.stoi[self.eos]]

    def decode(self, tokens):
        return " ".join(self.itos[t] for t in tokens if t > 1)


# -------------------------
# Paragraph Corpus
# -------------------------
corpus = """
wheat requires moderate irrigation and fertile soil for optimal growth
nitrogen improves crop yield and enhances leaf development in wheat plants
farmers monitor soil moisture levels to avoid water stress during dry seasons
maize needs frequent watering and proper sunlight to achieve maximum productivity
phosphorus supports root growth and early plant establishment in cereal crops
potassium improves disease resistance and strengthens plant cell walls
aphids damage wheat crops by sucking sap from young leaves and stems
integrated pest management helps reduce crop loss and improves sustainability
crop rotation enhances soil fertility and prevents pest population buildup
timely irrigation and balanced fertilizer application increase agricultural yield
soil testing helps farmers choose the correct nutrient management strategy
modern agriculture relies on data driven decisions and climate awareness
wheat growth depends on consistent water supply and balanced soil nutrients
nitrogen deficiency causes pale leaves and reduced biomass in cereal crops
excess irrigation can lead to root diseases and nutrient leaching in soil
maize plants benefit from early irrigation during vegetative growth stages
phosphorus availability is critical during seedling establishment and rooting
potassium application improves plant tolerance to heat and drought stress
pest infestations reduce crop productivity and affect grain quality
biological control methods reduce reliance on chemical pesticides
crop diversification lowers economic risk and improves farm resilience
efficient fertilizer management reduces environmental pollution and costs
soil organic matter improves water retention and microbial activity
precision agriculture tools optimize irrigation and nutrient placement
weather variability affects planting dates and crop performance
drought stress during flowering reduces grain formation in maize
balanced nutrition improves photosynthesis and energy transfer in plants
integrated soil fertility management sustains long term productivity
farmers adjust irrigation schedules based on rainfall and temperature
crop residues protect soil surface and reduce evaporation losses
healthy soil structure improves root penetration and nutrient uptake
early pest detection allows timely management interventions
sustainable farming practices improve yield stability over seasons
wheat varieties differ in water use efficiency and nutrient demand
maize hybrids respond differently to fertilizer application rates
soil pH influences nutrient availability and crop uptake
nutrient imbalance can limit growth even with adequate irrigation
monitoring crop health improves decision making and management accuracy
agricultural productivity depends on soil health and water management
improved seed quality enhances crop establishment and uniform growth
plant stress reduces resistance to insects and plant diseases
efficient water use increases yield under limited irrigation conditions
fertilizer timing affects nutrient uptake efficiency in crops
crop management practices influence long term soil sustainability
"""

tokenizer = DecoderTokenizer(corpus)
encoded = tokenizer.encode(corpus)
# print(encoded)
# -------------------------
# Build Training Data
# -------------------------
MAX_LEN = 128
inputs, targets = [], []

for i in range(len(encoded) - MAX_LEN):
    seq = encoded[i:i + MAX_LEN]
    inputs.append(seq[:-1])
    targets.append(seq[1:])

inputs = torch.tensor(inputs)
targets = torch.tensor(targets)
print(inputs)
print("\n")
print(targets)
# -------------------------
# #Model
# -------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = TinyGPT(
    vocab_size=len(tokenizer.stoi),
    d_model=128,
    layers=2,
    heads=4,
    max_len=MAX_LEN
).to(device)

inputs, targets = inputs.to(device), targets.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -------------------------
# Training
# -------------------------
EPOCHS = 50
losses = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()

    logits = model(inputs)  # (B, T, V)

    loss = criterion(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )

    loss.backward()
    optimizer.step()

    losses.append(loss.item())

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Loss {loss.item():.4f}")

plot_loss(losses)

# -------------------------
# Generation (BATCH = 1)
# -------------------------
model.eval()

start_text = "agriculture helps"
generated = torch.tensor([tokenizer.encode(start_text)[:-1]], device=device)
for _ in range(60):
    context = generated
    if context.size(1) > MAX_LEN:
        context = context[:, -MAX_LEN:]  # keep last MAX_LEN tokens

    with torch.no_grad():
        logits = model(context)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

    if next_token.item() == tokenizer.stoi["<eos>"]:
        break

    generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

print("\nGenerated text:")
print(tokenizer.decode(generated[0].tolist()))
