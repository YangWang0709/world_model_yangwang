from __future__ import annotations

import pytest

from eval.eval_teacher_student_gap import compute_teacher_student_gap


def test_compute_teacher_student_gap_and_ratio() -> None:
    metrics = compute_teacher_student_gap(teacher_future_mse=0.25, student_future_mse=0.5)

    assert metrics["student_teacher_gap"] == 0.25
    assert metrics["student_teacher_ratio"] == 2.0


def test_compute_teacher_student_gap_handles_zero_teacher_mse() -> None:
    metrics = compute_teacher_student_gap(teacher_future_mse=0.0, student_future_mse=0.0)

    assert metrics["student_teacher_gap"] == 0.0
    assert metrics["student_teacher_ratio"] == 0.0


def test_compute_teacher_student_gap_rejects_negative_mse() -> None:
    with pytest.raises(ValueError):
        compute_teacher_student_gap(teacher_future_mse=-1.0, student_future_mse=0.0)
