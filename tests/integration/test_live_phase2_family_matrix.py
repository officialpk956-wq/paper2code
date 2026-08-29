"""Opt-in real HTTP/LLM family matrix for the Phase 2 acceptance gate."""

import io
import os
import time

import fitz
import httpx
import pytest


_ENABLED = os.getenv("RUN_LIVE_FAMILY_MATRIX") == "1"
_BASE_URL = os.getenv("PAPER2CODE_LIVE_API_URL", "http://127.0.0.1:8010").rstrip("/")


CASES = [
    (
        "resnet-50-method-a.pdf",
        "resnet",
        "ResNet-50 uses a 7x7 convolution stem, then bottleneck residual stages with "
        "3, 4, 6, and 3 blocks. The stage widths are 64, 128, 256, and 512. Global "
        "average pooling feeds a 1000-class linear head.",
    ),
    (
        "deep-residual-network-b.pdf",
        "resnet",
        "Our residual network begins with a 64-channel stride-2 convolution and max pool. "
        "Four bottleneck stages repeat 3, 4, 6, and 3 times with identity shortcuts, "
        "followed by global average pooling and 1000-way classification.",
    ),
    (
        "u-net-segmentation-a.pdf",
        "unet",
        "U-Net has a contracting encoder with double 3x3 convolutions at 64, 128, 256, "
        "and 512 channels and max pooling. The decoder upsamples and concatenates matching "
        "encoder features through skip connections, producing a two-channel segmentation map.",
    ),
    (
        "u-net-biomedical-b.pdf",
        "unet",
        "The U-shaped segmentation network applies four convolutional downsampling blocks, "
        "a 1024-channel bottleneck, then four upsampling decoder blocks. Encoder activations "
        "are concatenated into the decoder and a 1x1 convolution predicts two classes.",
    ),
    (
        "vision-transformer-a.pdf",
        "vit",
        "Vision Transformer divides a 224x224 RGB image into 16x16 patches, projects each "
        "patch to a 768-dimensional token, prepends a class token, and applies 12 transformer "
        "encoder blocks with 12 attention heads before a 1000-class linear head.",
    ),
    (
        "vit-patch-model-b.pdf",
        "vit",
        "ViT uses patch embedding with patch size 16 and embedding dimension 768. Positional "
        "embeddings and a class token are processed by twelve multi-head self-attention and "
        "feed-forward blocks; the class representation predicts 1000 categories.",
    ),
    (
        "transformer-encoder-a.pdf",
        "transformer",
        "The Transformer encoder contains 6 layers. Every layer uses 8-head self-attention "
        "with d_model 512 and a feed-forward dimension of 2048, residual connections, and "
        "layer normalization. A linear projection produces 1000 output classes.",
    ),
    (
        "attention-is-all-you-need-b.pdf",
        "transformer",
        "Attention Is All You Need stacks six encoder layers with multi-head attention, "
        "eight heads, model width 512, and position-wise feed-forward width 2048. Token "
        "embeddings use positional encoding and the pooled sequence is classified.",
    ),
    (
        "stylegan-unsupported-a.pdf",
        "gan",
        "StyleGAN contains a mapping network and a generator that progressively synthesizes "
        "images from a learned constant using style-modulated convolutions. A discriminator "
        "with downsampling residual blocks distinguishes real and generated images.",
    ),
    (
        "ddpm-diffusion-unsupported-b.pdf",
        "diffusion",
        "A denoising diffusion probabilistic model trains a time-conditioned U-Net to predict "
        "Gaussian noise. A noise scheduler defines the forward process; sinusoidal timestep "
        "embeddings condition residual downsampling and upsampling blocks.",
    ),
]


def _pdf_bytes(title: str, methods: str) -> bytes:
    document = fitz.open()
    page = document.new_page()
    page.insert_textbox(
        fitz.Rect(48, 48, 548, 760),
        f"{title}\n\nMethods\n{methods}\n\nExperiments\nWe evaluate the architecture on its canonical task.",
        fontsize=11,
    )
    payload = document.tobytes()
    document.close()
    return payload


@pytest.mark.live
@pytest.mark.skipif(
    not _ENABLED,
    reason="requires RUN_LIVE_FAMILY_MATRIX=1 and a live API/Redis/Celery/LLM stack",
)
def test_phase2_ten_upload_family_matrix():
    with httpx.Client(base_url=_BASE_URL, timeout=90.0) as client:
        email = "phase2_family_matrix@example.com"
        password = "Phase2FamilyMatrix123!"
        client.post(
            "/api/auth/register",
            json={"email": email, "name": "Phase 2 Matrix", "password": password},
        )
        login = client.post(
            "/api/auth/login",
            data={"username": email, "password": password},
        )
        login.raise_for_status()
        token = login.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        results = []
        for filename, expected_family, excerpt in CASES:
            upload = client.post(
                "/api/papers/upload",
                headers=headers,
                files={"file": (filename, io.BytesIO(_pdf_bytes(filename, excerpt)), "application/pdf")},
                data={"terms_accepted": "true", "visibility": "private"},
            )
            upload.raise_for_status()
            queued = upload.json()
            assert queued["status"] == "pending"
            assert queued["paper_id"] is None

            deadline = time.monotonic() + 10 * 60
            while time.monotonic() < deadline:
                task_response = client.get(queued["poll_url"], headers=headers)
                task_response.raise_for_status()
                task = task_response.json()
                if task["status"] in ("completed", "failed"):
                    break
                time.sleep(1)
            else:
                pytest.fail(f"{filename} timed out")

            if task["status"] == "failed":
                results.append(
                    {
                        "filename": filename,
                        "expected_family": expected_family,
                        "family": None,
                        "generation_status": "failed",
                        "code_source": None,
                        "paper": None,
                        "error": task.get("error"),
                    }
                )
                continue

            result = task["result"]
            paper = client.get(f"/api/papers/{result['paper_id']}", headers=headers)
            paper.raise_for_status()
            results.append(
                {
                    "filename": filename,
                    "expected_family": expected_family,
                    "family": result.get("family"),
                    "generation_status": result.get("generation_status"),
                    "code_source": result.get("code_source"),
                    "paper": paper.json(),
                    "error": None,
                }
            )

    print(
        [
            {
                key: result.get(key)
                for key in (
                    "filename",
                    "expected_family",
                    "family",
                    "generation_status",
                    "code_source",
                    "error",
                )
            }
            for result in results
        ]
    )

    supported = [result for result in results if result["expected_family"] in {"resnet", "unet", "vit", "transformer"}]
    unsupported = [result for result in results if result["expected_family"] in {"gan", "diffusion"}]
    successful_supported = [
        result
        for result in supported
        if result["family"] == result["expected_family"]
        and result["generation_status"] == "success"
    ]

    assert len(successful_supported) >= 7, results
    assert all(result["family"] == result["expected_family"] for result in supported), results
    assert all(result["generation_status"] == "needs_review" for result in unsupported), results
    assert all(result["paper"] and result["paper"].get("verification_report") for result in results), results
