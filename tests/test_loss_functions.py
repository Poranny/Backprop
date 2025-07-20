import pytest
from src.loss_functions import MSE


def test_mse():
    input = [0, 1, 2, 3]
    labels = [0.5, 0.9, 1.9, 4]
    expected = (0.25 + 0.01 + 0.01 + 1) / 4.0
    result = MSE(input, labels)
    assert result == pytest.approx(expected)
