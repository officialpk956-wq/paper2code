import torch.nn as nn
from src.blocks_transformer import TransformerEncoderBlock


class TransformerBuilder(nn.Module):
    def __init__(self, schema):
        super().__init__()

        # ---- Required global params ----
        block_params = schema["block"]["params"]

        d_model = block_params["d_model"]
        vocab_size = schema["input"]["vocab_size"]
        num_classes = schema["output"]["num_classes"]

        # ---- Token embedding ----
        self.embedding = nn.Embedding(vocab_size, d_model)

        # ---- Encoder stack ----
        layers = []
        for stage in schema["stages"]:
            for _ in range(stage["num_blocks"]):
                layers.append(
                    TransformerEncoderBlock(
                        d_model=block_params["d_model"],
                        num_heads=block_params["num_heads"],
                        ffn_dim=block_params["ffn_dim"],
                        dropout=block_params.get("dropout", 0.1)
                    )
                )

        self.encoder = nn.Sequential(*layers)

        # ---- Classification head ----
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x):
        """
        x: (batch_size, sequence_length)
        """
        x = self.embedding(x)      # (B, T, d_model)
        x = self.encoder(x)        # (B, T, d_model)
        x = x.mean(dim=1)          # Global average pooling
        return self.head(x)
