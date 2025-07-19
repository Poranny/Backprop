def MSE(output: list, labels: list) -> float:
    mse = 0.0

    if len(labels) != len(output):
        raise ValueError("Number of labels does not match number of outputs")

    for i in range(len(output)):
        mse += (output[i] - labels[i]) ** 2

    mse /= len(output)

    return mse
