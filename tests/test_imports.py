"""Import checks for the Step 2 skeleton."""


def test_core_imports() -> None:
    import data.datasets  # noqa: F401
    import data.token_shards  # noqa: F401
    import data.video_transforms  # noqa: F401
    import encoders.dummy_video_encoder  # noqa: F401
    import encoders.text_encoder  # noqa: F401
    import encoders.videomae_wrapper  # noqa: F401
    import encoders.vjepa_wrapper  # noqa: F401
    import models.attention_selector  # noqa: F401
    import models.memory_bank  # noqa: F401
    import models.student_world_model  # noqa: F401
    import models.teacher_world_model  # noqa: F401
    import models.token_compressor  # noqa: F401
    import training.losses  # noqa: F401

