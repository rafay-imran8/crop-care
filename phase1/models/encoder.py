import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention
from utils.positional_encoding import PositionalEncoding

class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: (B, T, D)
        x = self.norm1(x + self.attn(x))
        x = self.norm2(x + self.ff(x))
        return x


class EncoderClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, layers=2, heads=4, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model) #embedding is being done here size of 128 per token
        self.positional = PositionalEncoding(d_model, max_len=128)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, heads) for _ in range(layers)
        ])

        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (B, T)
        """
        x = self.embedding(x)        # (B, T, D)
        x = self.positional(x)       # (B, T, D)

        for layer in self.layers:
            x = layer(x)

        pooled = x.mean(dim=1)       # (B, D)
        logits = self.classifier(pooled)  # (B, C)
        return logits
