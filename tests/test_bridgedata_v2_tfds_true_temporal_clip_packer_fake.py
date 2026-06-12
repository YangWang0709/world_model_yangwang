import torch

from data.bridgedata_v2_tfds_true_temporal_schema import pack_temporal_clip


def test_true_temporal_clip_packer_repeats_four_frames_to_sixteen_in_order():
    video = torch.arange(4, dtype=torch.float32).reshape(4, 1, 1, 1)
    packed = pack_temporal_clip(video, required_frames=16)
    assert list(packed.shape) == [16, 1, 1, 1]
    assert packed[:, 0, 0, 0].tolist() == [
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        1.0,
        1.0,
        1.0,
        2.0,
        2.0,
        2.0,
        2.0,
        3.0,
        3.0,
        3.0,
        3.0,
    ]


def test_true_temporal_clip_packer_keeps_sixteen_frame_context():
    video = torch.randn(16, 3, 2, 2)
    packed = pack_temporal_clip(video, required_frames=16)
    assert torch.equal(packed, video)
