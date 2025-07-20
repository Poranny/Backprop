import numpy as np


def MSE(output: list, labels: list) -> float:

    if len(labels) != len(output):
        raise ValueError("Number of labels does not match number of outputs")

    output = np.array(output)
    labels = np.array(labels)

    mse = np.mean((output - labels) ** 2)

    return float(mse)
