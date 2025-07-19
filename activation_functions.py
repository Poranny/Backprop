import math

def Identity (x : float, is_derivative : bool = False) -> float :
    if not is_derivative :
        return x
    else :
        return 1.0

def ReLU (x : float, is_derivative : bool = False) -> float :
    if not is_derivative:
        return max(0.0, x)
    else :
        return float(x > 0)

def Sigmoid (x : float, is_derivative : bool = False) -> float :
    sigmoid_val = 1.0 / (1.0 + math.exp(-x))

    if not is_derivative:
        return sigmoid_val
    else :
        return sigmoid_val * (1 - sigmoid_val)
