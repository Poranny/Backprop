from typing import Tuple, List

from neural_defs import NeuralNetwork
from loss_functions import MSE


def train_epoch(nn: NeuralNetwork, data: List[Tuple[List, List]], lr: float, log=False):

    total_loss = 0.0

    for row in data:
        inputs, expected_outputs = row

        nn.set_source_inputs(inputs)

        nn.calculate_output()

        output = nn.get_output()

        loss_fn = MSE

        loss = loss_fn(output, expected_outputs)

        if log:
            print(f"Loss {loss} at {row} -- given output {output}")

        total_loss += loss

        nn.backprop(expected_outputs, lr)

    return total_loss / len(data)
