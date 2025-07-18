import math

def Identity (x : float) -> float :
    return x

def ReLU (x : float) -> float :
    return max(0.0, x)

def Sigmoid (x : float) -> float :
    return 1.0 / (1.0 + math.exp(-x))

def BinaryStep (x : float) -> float :
    return float(x > 0)