"""
Transformer Model Builder for Paper2Code.

Provides a self-contained, executable PyTorch nn.Module for Transformer models.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.blocks_transformer import TransformerDecoderBlock, TransformerEncoderBlock


class TransformerModelBuilder(nn.Module):
    """Transformer encoder, plus a decoder stack when the schema asks for one.

    ``decoder_stages`` drives it. Without that key the model is exactly the
    encoder-only classifier it has always been; with it, the model becomes
    seq2seq and projects to the vocabulary instead of to class logits.
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
            block_params.get("ffn_dim") or block_params.get("hidden_size") or (self.d_model * 4)
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

        # Decoder stack -- only when the spec actually describes one.
        decoder_stages = schema.get("decoder_stages") or []
        decoder_layers = []
        for stage in decoder_stages:
            repeats = int(stage.get("repeats") or stage.get("num_blocks") or 1)
            for _ in range(max(1, repeats)):
                decoder_layers.append(
                    TransformerDecoderBlock(
                        d_model=self.d_model,
                        num_heads=self.num_heads,
                        ffn_dim=self.ffn_dim,
                        dropout=self.dropout,
                    )
                )
        self.decoder = nn.ModuleList(decoder_layers) if decoder_layers else None

        # Final LayerNorm, then either a vocabulary projection (seq2seq) or a
        # classification head (encoder-only).
        self.norm = nn.LayerNorm(self.d_model)
        if self.decoder is not None:
            self.output_proj = nn.Linear(self.d_model, self.vocab_size)
            self.fc = self.output_proj
        else:
            self.fc = nn.Linear(self.d_model, self.num_classes)

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """Token indices (B, T) or pre-embedded (B, T, D) -> (B, T, d_model)."""
        if x.dim() == 2:
            B, T = x.shape
            x = self.embedding(x) + self.pos_embedding[:, :T, :]
            return self.drop(x)
        B, T, D = x.shape
        if D != self.d_model:
            if not hasattr(self, "_in_proj"):
                self._in_proj = nn.Linear(D, self.d_model).to(x.device)
            x = self._in_proj(x)
        x = x + self.pos_embedding[:, :T, :]
        return self.drop(x)

    def forward(self, x: torch.Tensor, tgt: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass. Accepts either 2D token indices (B, T) or 3D embedded tensors (B, T, D).

        With a decoder, returns (B, T_tgt, vocab_size); ``tgt`` defaults to the
        source so a single-argument smoke test still works. Without one, the
        original encoder-only path is unchanged: pooled, then class logits.
        """
        if self.decoder is not None:
            memory = self.norm(self.encoder(self._embed(x)))
            y = self._embed(x if tgt is None else tgt)
            for block in self.decoder:
                y = block(y, memory)
            return self.output_proj(self.norm(y))

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
