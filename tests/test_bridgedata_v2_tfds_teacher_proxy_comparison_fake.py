import torch

from data.bridgedata_v2_tfds_teacher_proxy_comparison import pearson_corr, spearman_corr


def test_teacher_proxy_correlations_handle_high_and_low_cases():
    x = torch.arange(100, dtype=torch.float32)
    y = x.clone()
    z = torch.flip(x, dims=[0])
    assert pearson_corr(x, y) > 0.99
    assert spearman_corr(x, y) > 0.99
    assert pearson_corr(x, z) < -0.99
    assert spearman_corr(x, z) < -0.99


def test_teacher_proxy_correlations_do_not_crash_on_constant_labels():
    x = torch.ones(100)
    y = torch.arange(100, dtype=torch.float32)
    assert pearson_corr(x, y) == 0.0
    assert spearman_corr(x, y) == 0.0
