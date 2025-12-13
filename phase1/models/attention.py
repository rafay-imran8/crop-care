# phase1/models/attention.py
import torch
import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0
        self.d_k = d_model // heads
        self.heads = heads

        self.qkv = nn.Linear(d_model, d_model * 3)
        self.fc = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, D = x.shape

        qkv = self.qkv(x)               # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.heads, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.heads, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.heads, self.d_k).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = torch.softmax(scores, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.fc(out)
