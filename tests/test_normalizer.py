import numpy as np
import pytest
from src.normalizer import Normalizer

@pytest.mark.parametrize(("fit_data", "tested_num", "expected_num"), [(
        [
            ([0], [1]),
            ([1], [2]),
            ([2], [3]),
        ],
        [
            3
        ],
        [
            2.449
        ]
    )])
def test_normalization_input_num(fit_data, tested_num, expected_num):
    normalizer = Normalizer(fit_data)
    result = normalizer.transform_input_number(tested_num)

    assert result == pytest.approx(expected_num, rel=1e-2)


@pytest.mark.parametrize(("fit_data", "tested_data", "expected_data"), [(
        [
            ([0], [1]),
            ([1], [2]),
            ([2], [3]),
        ],
        [-1, 4, 5],
        [-2.449, 3.674, 4.899]
)])
def test_normalization_input_array(fit_data, tested_data, expected_data):
    normalizer = Normalizer(fit_data)
    result = normalizer.transform_inputs(tested_data)

    assert result == pytest.approx(expected_data, rel=1e-2)

@pytest.mark.parametrize(("fit_data", "tested_data", "expected_data"), [(
        [
            ([0], [5]),
            ([1], [3]),
            ([2], [20]),
        ],
        [8, 49, 59],
        [8.573, 58.788, 71.035]
)])
def test_normalization_output_array(fit_data, tested_data, expected_data):
    normalizer = Normalizer(fit_data)
    result = normalizer.transform_inputs(tested_data)

    assert result == pytest.approx(expected_data, rel=1e-2)

@pytest.mark.parametrize(("fit_data", "tested_data", "expected_data"), [(
        [
            ([10], [5]),
            ([-10], [3]),
            ([-20], [20]),
            ([-100], [111]),
        ],
        [
            ([10], [5]),
            ([-10], [3]),
            ([-20], [20]),
            ([-100], [111]),
        ],
        [
            ([0.956], [-0.668]),
            ([0.478], [-0.713]),
            ([0.239], [-0.331]),
            ([-1.673], [1.713]),
        ],
)])
def test_normalization_array(fit_data, tested_data, expected_data):
    normalizer = Normalizer(fit_data)
    result = normalizer.transform(tested_data)

    result_np = np.array([[x[0], y[0]] for x, y in result])
    expected_np = np.array([[x[0], y[0]] for x, y in expected_data])

    assert np.allclose(result_np, expected_np, rtol=1e-2)