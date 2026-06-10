"""V-JEPA wrapper placeholder.

Step 2 intentionally does not download or load V-JEPA weights.
"""


class VJEPAWrapper:
    def __init__(self, *args, **kwargs) -> None:
        self.config = {"status": "placeholder_only"}

    def encode(self, *args, **kwargs):
        raise RuntimeError("V-JEPA weights are not configured in Step 2.")

