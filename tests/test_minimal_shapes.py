"""Shape tests for the minimal token selection pipeline."""

import torch

from encoders.dummy_video_encoder import DummyVideoEncoder
from models.attention_selector import AttentionSelector, select_topk
from models.student_world_model import StudentWorldModel
from models.teacher_world_model import TeacherWorldModel
from models.token_compressor import TokenCompressor


def test_minimal_pipeline_shapes() -> None:
    batch_size = 2
    num_tokens = 196
    token_dim = 768
    topk = 40
    num_latents = 16
    latent_dim = 512

    video = torch.randn(batch_size, 4, 3, 224, 224)
    task_embedding = torch.zeros(batch_size, token_dim)

    encoder = DummyVideoEncoder(num_tokens=num_tokens, token_dim=token_dim)
    teacher = TeacherWorldModel(token_dim=token_dim)
    selector = AttentionSelector(token_dim=token_dim)
    compressor = TokenCompressor(token_dim=token_dim, num_latents=num_latents, latent_dim=latent_dim)
    student = StudentWorldModel(latent_dim=latent_dim, output_dim=token_dim)

    full_tokens = encoder(video)
    assert list(full_tokens.shape) == [batch_size, num_tokens, token_dim]

    scores = selector(full_tokens, task_embedding=task_embedding)
    assert list(scores.shape) == [batch_size, num_tokens]

    selected_tokens, indices = select_topk(full_tokens, scores, k=topk)
    assert list(selected_tokens.shape) == [batch_size, topk, token_dim]
    assert list(indices.shape) == [batch_size, topk]

    compressed_latents = compressor(selected_tokens)
    assert list(compressed_latents.shape) == [batch_size, num_latents, latent_dim]

    teacher_pred = teacher(full_tokens)
    student_pred = student(compressed_latents)
    assert list(teacher_pred.shape) == [batch_size, token_dim]
    assert list(student_pred.shape) == [batch_size, token_dim]

