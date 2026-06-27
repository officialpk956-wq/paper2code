import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_seq_len, d_model)
        self.LayerNorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.max_seq_len = max_seq_len

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        
        embeddings = words_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, is_causal=True, max_len=1024):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_model = d_model
        self.is_causal = is_causal
        
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        
        if self.is_causal:
            self.register_buffer("causal_mask", torch.tril(torch.ones(max_len, max_len)).view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.qkv(x).split(self.d_model, dim=2)
        q, k, v = [t.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2) for t in qkv]
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        if self.is_causal:
            mask = self.causal_mask[:, :, :T, :T] == 0
            att = att.masked_fill(mask, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_dim, is_causal=False):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, num_heads, is_causal=is_causal)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
