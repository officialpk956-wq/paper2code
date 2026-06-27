import torch.nn as nn
from core.blocks_bert_gpt import BertEmbeddings, TransformerBlock

class BertGPTBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()
        vocab_size = schema["vocab_size"]
        max_seq_len = schema["max_seq_len"]
        d_model = schema["d_model"]
        depth = schema["depth"]
        num_heads = schema["num_heads"]
        ffn_dim = schema["ffn_dim"]
        is_causal = schema["is_causal"]
        
        self.embeddings = BertEmbeddings(vocab_size, max_seq_len, d_model)
        
        blocks = []
        for _ in range(depth):
            blocks.append(TransformerBlock(d_model, num_heads, ffn_dim, is_causal))
        self.encoder = nn.Sequential(*blocks)
        
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids):
        x = self.embeddings(input_ids)
        x = self.encoder(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits
