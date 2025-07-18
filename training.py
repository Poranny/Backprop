from defs import NeuralNetwork


def train(nn : NeuralNetwork, data) :
    nn.set_source_inputs(data[0])

    nn.calculate_output()

    output = nn.get_output()

    loss_fn = MSE

    loss_fn(output, data[1])

def MSE (output : list, labels : list) -> float :
    mse = 0.0

    if len(labels) != len(output) :
        raise ValueError('Number of labels does not match number of outputs')

    for i in range(len(output)) :
        mse += (output[i] - labels[i])**2

    mse /= len(output)

    return mse