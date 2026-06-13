import torch

from data.bridgedata_v2_proxy_label_redesign_step39a import (
    build_coarse_grid_label_if_possible,
    build_coarse_index_bins_label,
    build_denoised_soft_topk_label,
    build_global_spatial_prior,
    build_global_spatial_prior_removed_residual_label,
    build_temporal_broadcast_label,
    build_temporal_broadcast_minmax_label,
    build_temporal_broadcast_rank_soft_label,
    build_temporal_only_label,
)


def test_step39a_fake_label_variants_have_expected_shapes_and_ranges():
    torch.manual_seed(42)
    importance = torch.rand(16, 392)
    temporal = build_temporal_only_label(importance)

    assert list(temporal["native_label"].shape) == [16]
    assert list(temporal["broadcast_label"].shape) == [16, 392]
    for label in [
        build_temporal_broadcast_label(importance),
        build_temporal_broadcast_minmax_label(importance),
        build_temporal_broadcast_rank_soft_label(importance),
        build_coarse_index_bins_label(importance, 49),
        build_coarse_index_bins_label(importance, 98),
        build_coarse_index_bins_label(importance, 196),
        build_coarse_grid_label_if_possible(importance, [14, 28], [7, 14]),
        build_coarse_grid_label_if_possible(importance, [14, 28], [7, 7]),
        build_denoised_soft_topk_label(importance, 256, 0.15, 0.02),
        build_denoised_soft_topk_label(importance, 512, 0.15, 0.02),
    ]:
        assert list(label.shape) == [16, 392]
        assert bool(torch.isfinite(label).all())


def test_step39a_global_prior_labels_have_expected_shape():
    importance = torch.rand(16, 392)
    prior = build_global_spatial_prior([{"importance": importance}, {"importance": importance * 0.9}])["prior"]

    label = build_global_spatial_prior_removed_residual_label(importance, prior)

    assert list(prior.shape) == [392]
    assert list(label.shape) == [16, 392]
    assert float(label.min()) >= 0.0
    assert float(label.max()) <= 1.0
