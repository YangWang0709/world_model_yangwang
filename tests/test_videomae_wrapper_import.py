"""Import-safety tests for the Step 9B VideoMAE wrapper."""

from __future__ import annotations

from encoders.videomae_wrapper import VideoMAEWrapper


def test_videomae_wrapper_imports_without_optional_weight_download() -> None:
    wrapper = VideoMAEWrapper(
        {
            "model_name_or_path": None,
            "allow_download": False,
            "local_files_only": True,
            "device": "cpu",
        }
    )

    assert isinstance(wrapper.is_available(), bool)
    assert wrapper.availability["available"] is False
    assert "reason" in wrapper.availability
    assert wrapper.availability["allow_download"] is False
    assert wrapper.availability["local_files_only"] is True
