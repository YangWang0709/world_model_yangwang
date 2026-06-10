"""Run the minimal Step 2 shape pipeline and write docs/SMOKE_TEST_REPORT.md."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from encoders.dummy_video_encoder import DummyVideoEncoder
from models.attention_selector import AttentionSelector, select_topk
from models.student_world_model import StudentWorldModel
from models.teacher_world_model import TeacherWorldModel
from models.token_compressor import TokenCompressor
from training.losses import budget_loss, distill_loss, future_latent_mse


REPORT_PATH = PROJECT_ROOT / "docs" / "SMOKE_TEST_REPORT.md"


def shape_text(tensor: torch.Tensor) -> str:
    return str(list(tensor.shape))


def run_smoke_test() -> tuple[bool, str]:
    torch.manual_seed(7)
    device = torch.device("cpu")

    batch_size = 2
    clip_len = 4
    channels = 3
    height = 224
    width = 224
    num_tokens = 196
    token_dim = 768
    topk_tokens = 40
    compressed_tokens = 16
    latent_dim = 512

    video = torch.randn(batch_size, clip_len, channels, height, width, device=device)
    task_embedding = torch.zeros(batch_size, token_dim, device=device)

    encoder = DummyVideoEncoder(num_tokens=num_tokens, token_dim=token_dim).to(device)
    teacher = TeacherWorldModel(token_dim=token_dim, hidden_dim=512).to(device)
    selector = AttentionSelector(token_dim=token_dim, hidden_dim=256).to(device)
    compressor = TokenCompressor(
        token_dim=token_dim,
        num_latents=compressed_tokens,
        latent_dim=latent_dim,
    ).to(device)
    student = StudentWorldModel(latent_dim=latent_dim, hidden_dim=512, output_dim=token_dim).to(device)

    with torch.no_grad():
        full_tokens = encoder(video)
        teacher_future_latent = teacher(full_tokens)
        scores = selector(full_tokens, task_embedding=task_embedding)
        selected_tokens, indices = select_topk(full_tokens, scores, k=topk_tokens)
        compressed_latents = compressor(selected_tokens)
        student_future_latent = student(compressed_latents)
        target_future_latent = teacher_future_latent.detach()
        future_loss = future_latent_mse(student_future_latent, target_future_latent)
        distill = distill_loss(student_future_latent, teacher_future_latent)
        budget = budget_loss(scores, target_keep_ratio=0.2)

    expected = {
        "full_tokens": [batch_size, num_tokens, token_dim],
        "teacher_future_latent": [batch_size, token_dim],
        "scores": [batch_size, num_tokens],
        "selected_tokens": [batch_size, topk_tokens, token_dim],
        "indices": [batch_size, topk_tokens],
        "compressed_latents": [batch_size, compressed_tokens, latent_dim],
        "student_future_latent": [batch_size, token_dim],
    }
    actual = {
        "full_tokens": list(full_tokens.shape),
        "teacher_future_latent": list(teacher_future_latent.shape),
        "scores": list(scores.shape),
        "selected_tokens": list(selected_tokens.shape),
        "indices": list(indices.shape),
        "compressed_latents": list(compressed_latents.shape),
        "student_future_latent": list(student_future_latent.shape),
    }
    pass_flag = actual == expected

    lines = [
        "# Minimal Pipeline Smoke Test Report",
        "",
        "| Item | Value |",
        "| --- | --- |",
        f"| Device | `{device}` |",
        f"| dummy video shape | `{[batch_size, clip_len, channels, height, width]}` |",
        f"| full tokens shape | `{actual['full_tokens']}` |",
        f"| teacher output shape | `{actual['teacher_future_latent']}` |",
        f"| scores shape | `{actual['scores']}` |",
        f"| selected tokens shape | `{actual['selected_tokens']}` |",
        f"| topk indices shape | `{actual['indices']}` |",
        f"| compressed latents shape | `{actual['compressed_latents']}` |",
        f"| student output shape | `{actual['student_future_latent']}` |",
        f"| future loss | `{future_loss.item():.8f}` |",
        f"| distill loss | `{distill.item():.8f}` |",
        f"| budget loss | `{budget.item():.8f}` |",
        f"| SMOKE_TEST_PASS | `{str(pass_flag).lower()}` |",
        "",
        f"SMOKE_TEST_PASS = {str(pass_flag).lower()}",
    ]
    return pass_flag, "\n".join(lines) + "\n"


def main() -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pass_flag, report = run_smoke_test()
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    if not pass_flag:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

