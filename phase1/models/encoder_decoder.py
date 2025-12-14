import torch
import torch.nn as nn
from .attention import MultiHeadSelfAttention, MultiHeadCrossAttention

class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden=256):
        super().__init__()
        self.mha = MultiHeadSelfAttention(d_model, heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.mha(x, mask))
        x = self.ln2(x + self.ff(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, heads, ff_hidden=256):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, heads)
        self.cross_attn = MultiHeadCrossAttention(d_model, heads)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_hidden),
            nn.ReLU(),
            nn.Linear(ff_hidden, d_model)
        )

    def forward(self, x, enc_out, self_mask=None, cross_mask=None):
        x = self.ln1(x + self.self_attn(x, self_mask))
        x = self.ln2(x + self.cross_attn(x, enc_out, cross_mask))
        x = self.ln3(x + self.ff(x))
        return x

class EncoderDecoderTransformer(nn.Module):
    def __init__(self, enc_vocab_size, dec_vocab_size, d_model=128, enc_layers=2, dec_layers=2, heads=4, max_len=128):
        super().__init__()
        self.enc_token_emb = nn.Embedding(enc_vocab_size, d_model)
        self.dec_token_emb = nn.Embedding(dec_vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.enc_blocks = nn.ModuleList([EncoderBlock(d_model, heads) for _ in range(enc_layers)])
        self.dec_blocks = nn.ModuleList([DecoderBlock(d_model, heads) for _ in range(dec_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, dec_vocab_size)
        self.max_len = max_len

    def forward(self, enc_input, dec_input):
        B, T_enc = enc_input.shape
        B, T_dec = dec_input.shape

        enc_pos = torch.arange(T_enc, device=enc_input.device).unsqueeze(0)
        dec_pos = torch.arange(T_dec, device=dec_input.device).unsqueeze(0)

        enc_x = self.enc_token_emb(enc_input) + self.pos_emb(enc_pos)
        dec_x = self.dec_token_emb(dec_input) + self.pos_emb(dec_pos)

        # Encoder forward
        for block in self.enc_blocks:
            enc_x = block(enc_x)

        # Decoder forward with masks
        self_mask = torch.tril(torch.ones(T_dec, T_dec, device=dec_input.device)).unsqueeze(0).unsqueeze(0)
        cross_mask = None  # no masking for cross-attention
        for block in self.dec_blocks:
            dec_x = block(dec_x, enc_x, self_mask, cross_mask)

        dec_x = self.ln_f(dec_x)
        logits = self.fc_out(dec_x)  # (B, T_dec, dec_vocab_size)
        return logits
