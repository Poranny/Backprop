import numpy as np
import matplotlib.pyplot as plt
from activation_functions import Identity, Sigmoid, ReLU

def visualize_nn(weights, bias, activation):

    w = np.array(weights, dtype=float)
    if w.ndim > 1:
        w = w.flatten()

    if len(w) == 0:
        raise ValueError("At least one weight is required")

    if len(w) == 1:
        x = np.linspace(-100, 100, 400)
        z = np.array([activation(w[0] * xi + bias) for xi in x])
        plt.figure()
        plt.plot(x, z)
        plt.title("Activation Function Output (1D)")
        plt.xlabel("Input x")
        plt.ylabel("Output z")
        plt.grid(True)
        plt.show()

    else :
        w1, w2 = w[0], w[1]
        x = np.linspace(-100, 100, 200)
        y = np.linspace(-100, 100, 200)
        X, Y = np.meshgrid(x, y)
        Z = np.vectorize(lambda xi, yi: activation(w1 * xi + w2 * yi + bias))(X, Y)

        plt.figure()
        plt.title("Activation Function Output (2D)")
        plt.xlabel("Input x")
        plt.ylabel("Input y")
        plt.show()