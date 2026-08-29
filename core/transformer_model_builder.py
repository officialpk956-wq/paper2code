"""
Transformer Model Builder for Paper2Code.

Provides a self-contained, executable PyTorch nn.Module for Transformer models.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.blocks_transformer import TransformerEncoderBlock


class TransformerModelBuilder(nn.Module):
    """
    Standard Transformer Encoder architecture for classification / sequence processing.
    """

    def __init__(self, schema: dict):
        super().__init__()

        stem_params = (schema.get("stem") or {}).get("params") or {}
        block_params = (schema.get("block") or {}).get("params") or {}
        input_params = schema.get("input") or {}
        output_params = schema.get("output") or {}

        self.d_model = int(
            stem_params.get("d_model")
            or block_params.get("d_model")
            or block_params.get("embed_dim")
            or 512
        )
        self.num_heads = int(block_params.get("num_heads") or 8)
        self.ffn_dim = int(
            block_params.get("ffn_dim")
            or block_params.get("hidden_size")
            or (self.d_model * 4)
        )
        self.dropout = float(block_params.get("dropout") or 0.1)

        self.vocab_size = int(input_params.get("vocab_size") or 10000)
        self.max_seq_len = int(input_params.get("max_seq_len") or 512)
        self.num_classes = int(output_params.get("num_classes") or 1000)

        # Token + Positional Embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.max_seq_len, self.d_model))
        self.drop = nn.Dropout(self.dropout)

        # Transformer Encoder Blocks
        layers = []
        stages = schema.get("stages") or [{"repeats": 6}]
        for stage in stages:
            repeats = int(stage.get("repeats") or stage.get("num_blocks") or 1)
            for _ in range(max(1, repeats)):
                layers.append(
                    TransformerEncoderBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        ffn_dim=self.ffn_dim,
                        dropout=self.dropout,
                    )
                )
        self.encoder = nn.Sequential(*layers)

        # Final LayerNorm and Classification Head
        self.norm = nn.LayerNorm(self.d_model)
        self.fc = nn.Linear(self.d_model, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Accepts either 2D token indices (B, T) or 3D embedded tensors (B, T, D).
        """
        if x.dim() == 2:
            # Token index input: (B, T)
            B, T = x.shape
            x = self.embedding(x) + self.pos_embedding[:, :T, :]
            x = self.drop(x)
        elif x.dim() == 3:
            # Pre-embedded input: (B, T, D)
            B, T, D = x.shape
            if D != self.d_model:
                # Project if dimensions differ
                if not hasattr(self, "_in_proj"):
                    self._in_proj = nn.Linear(D, self.d_model).to(x.device)
                x = self._in_proj(x)
            x = x + self.pos_embedding[:, :T, :]
            x = self.drop(x)

        # Run through transformer encoder
        x = self.encoder(x)
        x = self.norm(x)

        # Sequence pooling: average across sequence length (dim=1)
        x = x.mean(dim=1)

        # Final classification head
        return self.fc(x)
