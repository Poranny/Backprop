import math


def Identity(x: float, is_derivative: bool = False) -> float:
    if not is_derivative:
        return x
    else:
        return 1.0


def ReLU(x: float, is_derivative: bool = False) -> float:
    if not is_derivative:
        return max(0.0, x)
    else:
        return float(x > 0)


def Sigmoid(x: float, is_derivative: bool = False) -> float:
    sigmoid_val = 1.0 / (1.0 + math.exp(-x))

    if not is_derivative:
        return sigmoid_val
    else:
        return sigmoid_val * (1 - sigmoid_val)


def Tanh(x: float, is_derivative: bool = False) -> float:
    tanh_val = math.tanh(x)
    if not is_derivative:
        return tanh_val
    else:
        return 1.0 - tanh_val**2
