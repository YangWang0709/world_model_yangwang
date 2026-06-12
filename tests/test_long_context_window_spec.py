import pytest

from data.long_context_window_spec import LongContextWindowSpec


def test_window_spec_returns_empty_when_trajectory_is_too_short():
    spec = LongContextWindowSpec(context_len=16, current_len=4, future_len=4)
    assert spec.generate_window_indices(total_frames=23) == []


def test_window_spec_generates_contiguous_context_current_future_indices():
    spec = LongContextWindowSpec(context_len=8, current_len=4, future_len=4, stride=4)
    windows = spec.generate_window_indices(total_frames=16)
    assert len(windows) == 1
    first = windows[0]
    assert first["context_frame_indices"] == list(range(0, 8))
    assert first["current_frame_indices"] == list(range(8, 12))
    assert first["future_frame_indices"] == list(range(12, 16))
    assert set(first["context_frame_indices"]).isdisjoint(first["current_frame_indices"])
    assert set(first["current_frame_indices"]).isdisjoint(first["future_frame_indices"])


@pytest.mark.parametrize("context_len", [8, 16, 32])
def test_window_spec_supports_required_context_lengths(context_len):
    spec = LongContextWindowSpec(context_len=context_len, current_len=4, future_len=8, stride=8)
    windows = spec.generate_window_indices(total_frames=context_len + 12)
    assert windows
    assert len(windows[0]["context_frame_indices"]) == context_len
    assert len(windows[0]["current_frame_indices"]) == 4
    assert len(windows[0]["future_frame_indices"]) == 8


def test_window_spec_supports_frame_rate_subsample():
    spec = LongContextWindowSpec(context_len=8, current_len=4, future_len=4, frame_rate_subsample=2)
    first = spec.generate_window_indices(total_frames=31)[0]
    assert first["context_frame_indices"][:3] == [0, 2, 4]
    assert first["future_frame_indices"][-1] == 30
