# model_utils.py
import math
import json
import torch
import torch.nn as nn
import gdown
import os  # new

# === Model Components (same as training) ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=128):  # must match training
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_k = d_model // num_heads
        self.h = num_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)
        def shape(x, linear):
            x = linear(x)
            return x.view(B, -1, self.h, self.d_k).transpose(1, 2)
        q = shape(query, self.linear_q)
        k = shape(key, self.linear_k)
        v = shape(value, self.linear_v)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.linear_out(x)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, heads, dropout)
        self.src_attn = MultiHeadAttention(d_model, heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_mask))
        x = self.norm2(x + self.src_attn(x, memory, memory, src_mask))
        x = self.norm3(x + self.ff(x))
        return x

class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embed(src) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout, pad_idx):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer(d_model, heads, d_ff, dropout) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, src_mask=None, tgt_mask=None):
        x = self.embed(tgt) * math.sqrt(self.embed.embedding_dim)
        x = self.pos(x)
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, pad_idx, d_model=256, N_enc=2, N_dec=2, heads=4, d_ff=512, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, N_enc, heads, d_ff, dropout, pad_idx)
        self.decoder = Decoder(vocab_size, d_model, N_dec, heads, d_ff, dropout, pad_idx)
        self.generator = nn.Linear(d_model, vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        out = self.decoder(tgt, memory, src_mask, tgt_mask)
        return self.generator(out)

# === Utility functions ===
def make_src_mask(src, pad_idx):
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def make_tgt_mask(tgt, pad_idx):
    pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_len = tgt.size(1)
    subsequent_mask = torch.triu(torch.ones((1, seq_len, seq_len), device=tgt.device), diagonal=1).bool()
    return pad_mask & (~subsequent_mask.unsqueeze(0))

def load_model(model_path, vocab_path, device):
    # If model file not local yet, download from Drive
    if not os.path.exists(model_path):
        # Google Drive file ID parsed from your share link
        # link: https://drive.google.com/file/d/17iK4WWzl36iefp9om3fnYRxHxum4mlP-/view?usp=drive_link
        file_id = "17iK4WWzl36iefp9om3fnYRxHxum4mlP-"
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"Downloading model from Drive: {url} → {model_path}")
        gdown.download(url, model_path, quiet=False)

    # Load vocab
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    stoi = vocab['stoi']
    itos = vocab['itos']
    pad_idx = stoi['<pad>']

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model = TransformerSeq2Seq(len(itos), pad_idx)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, stoi, itos

def encode_text(text, stoi, max_len=128):
    tokens = text.split()
    tokens = ['<bos>'] + tokens[:max_len-2] + ['<eos>']
    return [stoi.get(t, stoi['<unk>']) for t in tokens]

def decode_ids(ids, itos):
    words = []
    for i in ids:
        w = itos[i]
        if w == '<eos>':
            break
        if w not in ['<bos>', '<pad>']:
            words.append(w)
    return ' '.join(words)

def greedy_decode(model, src, stoi, itos, device, max_len=60):
    src_mask = make_src_mask(src, stoi['<pad>'])
    with torch.no_grad():
        memory = model.encoder(src, src_mask)
        ys = torch.full((src.size(0), 1), stoi['<bos>'], dtype=torch.long, device=device)
        for _ in range(max_len):
            tgt_mask = make_tgt_mask(ys, stoi['<pad>'])
            out = model.decoder(ys, memory, src_mask, tgt_mask)
            logits = model.generator(out)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
        return ys
