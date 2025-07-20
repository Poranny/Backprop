import pytest
from src.activation_functions import *

def test_sigmoid():
    input = 1.0
    expected = 0.731
    result = Sigmoid(input)

    assert result == pytest.approx(expected, 1e-2)


def test_identity():
    input = 1.0
    expected = 1.0
    result = Identity(input)

    assert result == pytest.approx(expected, 1e-2)


def test_tanh():
    input = 1.0
    expected = 0.762
    result = Tanh(input)

    assert result == pytest.approx(expected, 1e-2)


def test_relu():
    input = 1.0
    expected = 1.0
    result = ReLU(input)

    assert result == pytest.approx(expected, 1e-2)