import os
import sys
from sqlalchemy.orm import Session

# Add the project root to python path so we can import from backend
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.database import SessionLocal
from backend.models import Paper, PaperChallenge, PaperChallengePart

def seed():
    db = SessionLocal()

    papers = {
        "attention": db.query(Paper).filter(Paper.title == "Attention Is All You Need").first(),
        "resnet": db.query(Paper).filter(Paper.title == "Deep Residual Learning for Image Recognition").first(),
        "bert": db.query(Paper).filter(Paper.title == "BERT: Pre-training of Deep Bidirectional Transformers").first()
    }

    if not papers["attention"]:
        print("Warning: Paper 'Attention Is All You Need' not found, creating dummy...")
        papers["attention"] = Paper(title="Attention Is All You Need")
        db.add(papers["attention"])
        db.commit()
        db.refresh(papers["attention"])

    if not papers["resnet"]:
        print("Warning: Paper 'Deep Residual Learning for Image Recognition' not found, creating dummy...")
        papers["resnet"] = Paper(title="Deep Residual Learning for Image Recognition")
        db.add(papers["resnet"])
        db.commit()
        db.refresh(papers["resnet"])

    if not papers["bert"]:
        print("Warning: Paper 'BERT: Pre-training of Deep Bidirectional Transformers' not found, creating dummy...")
        papers["bert"] = Paper(title="BERT: Pre-training of Deep Bidirectional Transformers")
        db.add(papers["bert"])
        db.commit()
        db.refresh(papers["bert"])

    # ---------------------------------------------------------
    # Paper 1: Attention
    # ---------------------------------------------------------
    challenge_title_1 = "Implement the Transformer Attention Mechanism"
    c1 = db.query(PaperChallenge).filter_by(paper_id=papers["attention"].id, title=challenge_title_1).first()
    if not c1:
        c1 = PaperChallenge(paper_id=papers["attention"].id, title=challenge_title_1, is_published=True, order_idx=1)
        db.add(c1)
        db.commit()
        db.refresh(c1)

        # Part 1
        p1_1 = PaperChallengePart(
            challenge_id=c1.id,
            order_idx=1,
            title="Scaled Dot-Product Attention",
            description_md="Implement scaled dot-product attention.",
            setup_code="import torch; import torch.nn.functional as F; import math",
            starter_code='''def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Args:
        Q, K, V: tensors of shape (batch, heads, seq_len, d_k)
        mask: optional boolean tensor, True where attention should be blocked
    Returns:
        output tensor of shape (batch, heads, seq_len, d_k)
    """
    pass''',
            test_code='''Q = torch.randn(2, 4, 6, 16)
K = torch.randn(2, 4, 6, 16)
V = torch.randn(2, 4, 6, 16)
out = scaled_dot_product_attention(Q, K, V)
assert out is not None, "Function returned None"
assert out.shape == (2, 4, 6, 16), f"Wrong shape: {out.shape}"
assert not torch.isnan(out).any(), "Output has NaN"
expected = F.softmax(Q @ K.transpose(-2,-1) / math.sqrt(16), dim=-1) @ V
assert torch.allclose(out, expected, atol=1e-5), "Values incorrect"
print("All tests passed.")''',
            xp_reward=75,
            unlock_requires_part_id=None
        )
        db.add(p1_1)
        db.commit()
        db.refresh(p1_1)

        # Part 2
        p1_2 = PaperChallengePart(
            challenge_id=c1.id,
            order_idx=2,
            title="Multi-Head Attention",
            description_md="Implement the MultiHeadAttention module.",
            setup_code="import torch; import torch.nn as nn; import math",
            starter_code='''class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        # d_model must be divisible by num_heads
        pass

    def forward(self, Q, K, V, mask=None):
        # Q, K, V shape: (batch, seq_len, d_model)
        # Returns: (batch, seq_len, d_model)
        pass''',
            test_code='''mha = MultiHeadAttention(d_model=32, num_heads=4)
x = torch.randn(2, 6, 32)
out = mha(x, x, x)
assert out is not None, "Forward returned None"
assert out.shape == (2, 6, 32), f"Wrong shape: {out.shape}"
assert not torch.isnan(out).any(), "Output has NaN"
print("All tests passed.")''',
            xp_reward=100,
            unlock_requires_part_id=p1_1.id
        )
        db.add(p1_2)
        db.commit()
        db.refresh(p1_2)

        # Part 3
        p1_3 = PaperChallengePart(
            challenge_id=c1.id,
            order_idx=3,
            title="Positional Encoding",
            description_md="Implement positional encoding.",
            setup_code="import torch; import torch.nn as nn; import math",
            starter_code='''class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Create a fixed positional encoding matrix
        pass

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        # Add positional encodings to x
        pass''',
            test_code='''pe = PositionalEncoding(d_model=16, max_len=100)
x = torch.zeros(2, 10, 16)
out = pe(x)
assert out is not None, "Forward returned None"
assert out.shape == (2, 10, 16), f"Wrong shape: {out.shape}"
assert not torch.equal(out, x), "Positional encoding was not added"
assert not torch.isnan(out).any(), "Output has NaN"
print("All tests passed.")''',
            xp_reward=75,
            unlock_requires_part_id=p1_2.id
        )
        db.add(p1_3)
        db.commit()

    # ---------------------------------------------------------
    # Paper 2: ResNet
    # ---------------------------------------------------------
    challenge_title_2 = "Implement the ResNet Residual Block"
    c2 = db.query(PaperChallenge).filter_by(paper_id=papers["resnet"].id, title=challenge_title_2).first()
    if not c2:
        c2 = PaperChallenge(paper_id=papers["resnet"].id, title=challenge_title_2, is_published=True, order_idx=1)
        db.add(c2)
        db.commit()
        db.refresh(c2)

        p2_1 = PaperChallengePart(
            challenge_id=c2.id,
            order_idx=1,
            title="Basic Residual Block",
            description_md="Implement the basic residual block.",
            setup_code="import torch; import torch.nn as nn",
            starter_code='''class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # Two 3x3 conv layers with batch norm
        # Shortcut connection (identity or 1x1 conv if dims change)
        pass

    def forward(self, x):
        pass''',
            test_code='''block = BasicBlock(64, 64)
x = torch.randn(2, 64, 32, 32)
out = block(x)
assert out is not None, "Forward returned None"
assert out.shape == (2, 64, 32, 32), f"Wrong shape: {out.shape}"
assert not torch.isnan(out).any(), "Output has NaN"
block_down = BasicBlock(64, 128, stride=2)
out2 = block_down(x)
assert out2.shape == (2, 128, 16, 16), f"Downsampled shape wrong: {out2.shape}"
print("All tests passed.")''',
            xp_reward=100,
            unlock_requires_part_id=None
        )
        db.add(p2_1)
        db.commit()
        db.refresh(p2_1)

        p2_2 = PaperChallengePart(
            challenge_id=c2.id,
            order_idx=2,
            title="Bottleneck Block",
            description_md="Implement the bottleneck block.",
            setup_code="import torch; import torch.nn as nn",
            starter_code='''class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        # 1x1 conv → 3x3 conv → 1x1 conv pattern
        # Shortcut connection
        pass

    def forward(self, x):
        pass''',
            test_code='''block = Bottleneck(256, 64)
x = torch.randn(2, 256, 14, 14)
out = block(x)
assert out is not None, "Forward returned None"
assert out.shape == (2, 256, 14, 14), f"Wrong shape: {out.shape}"
assert not torch.isnan(out).any(), "Output has NaN"
print("All tests passed.")''',
            xp_reward=100,
            unlock_requires_part_id=p2_1.id
        )
        db.add(p2_2)
        db.commit()

    # ---------------------------------------------------------
    # Paper 3: BERT
    # ---------------------------------------------------------
    challenge_title_3 = "Implement BERT Building Blocks"
    c3 = db.query(PaperChallenge).filter_by(paper_id=papers["bert"].id, title=challenge_title_3).first()
    if not c3:
        c3 = PaperChallenge(paper_id=papers["bert"].id, title=challenge_title_3, is_published=True, order_idx=1)
        db.add(c3)
        db.commit()
        db.refresh(c3)

        p3_1 = PaperChallengePart(
            challenge_id=c3.id,
            order_idx=1,
            title="BERT Embeddings",
            description_md="Implement the embeddings layer of BERT.",
            setup_code="import torch; import torch.nn as nn",
            starter_code='''class BERTEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, num_segments=2):
        super().__init__()
        # Token embeddings + positional embeddings + segment embeddings
        # Layer norm + dropout
        pass

    def forward(self, input_ids, segment_ids=None):
        # input_ids: (batch, seq_len)
        # Returns: (batch, seq_len, d_model)
        pass''',
            test_code='''emb = BERTEmbeddings(vocab_size=1000, d_model=32)
ids = torch.randint(0, 1000, (2, 10))
seg = torch.zeros(2, 10, dtype=torch.long)
out = emb(ids, seg)
assert out is not None, "Forward returned None"
assert out.shape == (2, 10, 32), f"Wrong shape: {out.shape}"
assert not torch.isnan(out).any(), "Output has NaN"
print("All tests passed.")''',
            xp_reward=75,
            unlock_requires_part_id=None
        )
        db.add(p3_1)
        db.commit()

    db.close()
    print("Seed complete.")

if __name__ == "__main__":
    seed()
