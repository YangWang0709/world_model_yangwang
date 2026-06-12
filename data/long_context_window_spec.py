"""Window generation utilities for long-context robot trajectories."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LongContextWindowSpec:
    context_len: int
    current_len: int
    future_len: int
    stride: int = 1
    min_trajectory_len: int | None = None
    frame_rate_subsample: int = 1

    def validate(self) -> "LongContextWindowSpec":
        fields = {
            "context_len": self.context_len,
            "current_len": self.current_len,
            "future_len": self.future_len,
            "stride": self.stride,
            "frame_rate_subsample": self.frame_rate_subsample,
        }
        for name, value in fields.items():
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if self.min_trajectory_len is not None and self.min_trajectory_len <= 0:
            raise ValueError("min_trajectory_len must be positive when provided")
        return self

    @property
    def sampled_window_len(self) -> int:
        return self.context_len + self.current_len + self.future_len

    @property
    def required_total_frames(self) -> int:
        sampled_required = (self.sampled_window_len - 1) * self.frame_rate_subsample + 1
        if self.min_trajectory_len is None:
            return sampled_required
        return max(sampled_required, self.min_trajectory_len)

    def generate_window_indices(self, total_frames: int) -> list[dict[str, list[int]]]:
        self.validate()
        if total_frames < self.required_total_frames:
            return []

        windows: list[dict[str, list[int]]] = []
        last_start = total_frames - ((self.sampled_window_len - 1) * self.frame_rate_subsample + 1)
        start = 0
        while start <= last_start:
            sampled = [start + i * self.frame_rate_subsample for i in range(self.sampled_window_len)]
            context_end = self.context_len
            current_end = context_end + self.current_len
            windows.append(
                {
                    "context_frame_indices": sampled[:context_end],
                    "current_frame_indices": sampled[context_end:current_end],
                    "future_frame_indices": sampled[current_end:],
                }
            )
            start += self.stride
        return windows


def window_spec_from_dict(data: dict[str, int]) -> LongContextWindowSpec:
    return LongContextWindowSpec(
        context_len=int(data["context_len"]),
        current_len=int(data["current_len"]),
        future_len=int(data["future_len"]),
        stride=int(data.get("stride", 1)),
        min_trajectory_len=data.get("min_trajectory_len"),
        frame_rate_subsample=int(data.get("frame_rate_subsample", 1)),
    )
