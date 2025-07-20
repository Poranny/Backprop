from typing import Tuple, List

from src.neural_defs import NeuralNetwork
from src.loss_functions import MSE
from src.visualize_nn import visualize_full_nn
from src.normalizer import Normalizer


def train(
    nn: NeuralNetwork,
    data: List[Tuple[List, List]],
    iters: int,
    lr: float,
    normalizer: Normalizer = None,
    log=False,
):
    if normalizer is not None:
        norm_data = normalizer.transform(data)
    else:
        norm_data = data

    for epoch in range(iters):
        avg_loss = train_epoch(nn, norm_data, lr)

        if log:
            if epoch % (iters / 100) == 0:
                print(f"Epoch {epoch} avg loss: {avg_loss}")

            if epoch % (iters / 10) == 0:
                visualize_full_nn(
                    nn, x_range=(-10, 10), y_range=(-10, 10), normalizer=normalizer
                )


def train_epoch(nn: NeuralNetwork, data: List[Tuple[List, List]], lr: float):
    total_loss = 0.0

    for row in data:
        inputs, expected_outputs = row

        nn.set_source_inputs(inputs)

        nn.calculate_output()

        output = nn.get_output()

        loss_fn = MSE

        loss = loss_fn(output, expected_outputs)

        total_loss += loss

        nn.backprop(expected_outputs, lr)

    return total_loss / len(data)
