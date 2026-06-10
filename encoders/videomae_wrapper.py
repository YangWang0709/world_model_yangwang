"""VideoMAE wrapper placeholder.

Step 2 intentionally does not download or load VideoMAE weights.
"""


class VideoMAEWrapper:
    def __init__(self, *args, **kwargs) -> None:
        self.config = {"status": "placeholder_only"}

    def encode(self, *args, **kwargs):
        raise RuntimeError("VideoMAE weights are not configured in Step 2.")

