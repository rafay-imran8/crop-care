import torch
import torch.nn as nn
from models.attention import MultiHeadSelfAttention


class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()

        self.attn = MultiHeadSelfAttention(d_model, heads)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(),
            nn.Linear(4 * d_model, d_model)
        )

        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ff(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model, layers, heads, max_len):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)

        self.blocks = nn.ModuleList([
            DecoderBlock(d_model, heads) for _ in range(layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, vocab_size)
        self.max_len = max_len

    def forward(self, x):
        # x MUST be (B, T)
        B, T = x.shape
        assert T <= self.max_len, "Sequence length exceeds max_len"

        positions = torch.arange(T, device=x.device).unsqueeze(0)
        x = self.token_emb(x) + self.pos_emb(positions)

        # causal mask (shared across batch)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        mask = mask.unsqueeze(0).unsqueeze(0)
        # print(mask.shape)
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)
        return self.fc(x)  # (B, T, vocab)
