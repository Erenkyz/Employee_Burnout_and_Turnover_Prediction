import numpy as np
import pytest

from workforce_intelligence.model_stability import (
    _expected_calibration_error, evaluate_stability,
)
from test_model_validation import sample_data


def test_stability_is_deterministic():
    frame = sample_data(100)
    first = evaluate_stability(frame, seeds=(42, 43))
    second = evaluate_stability(frame, seeds=(42, 43))
    assert first == second
    assert len(first["runs"]) == 2
    assert first["runs"][0]["split_sizes"] == {
        "train": 60, "validation": 20, "test": 20,
    }
    assert 0 <= first["summary"]["ece_10_bins"]["mean"] <= 1


def test_ece_for_perfect_predictions():
    assert _expected_calibration_error(
        np.array([0, 1]), np.array([0.0, 1.0])
    ) == pytest.approx(0)


def test_duplicate_seeds_rejected():
    with pytest.raises(ValueError, match="unique seed"):
        evaluate_stability(sample_data(100), seeds=(42, 42))
