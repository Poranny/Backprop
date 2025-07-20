import numpy as np
import matplotlib.pyplot as plt
from src.testing import predict_with_normalization


def visualize_full_nn(
    neural_net, normalizer, x_range=(-1, 1), y_range=(-1, 1), num_points=200
):

    input_dim = neural_net.sourceLayer.layer_size
    output_dim = len(neural_net.layers[-1].neurons)

    if input_dim not in (1, 2):
        raise ValueError("Visualization only supports networks with 1D or 2D inputs.")

    if input_dim == 1:
        xs = np.linspace(x_range[0], x_range[1], num_points)
        for out_idx in range(output_dim):
            ys = []
            for x in xs:
                outputs = predict_with_normalization(neural_net, normalizer, [x])
                ys.append(outputs[out_idx])
            plt.figure()
            plt.plot(xs, ys)
            plt.title(f"Fully connected NN - Output Neuron {out_idx}")
            plt.xlabel("Input x")
            plt.ylabel(f"Output y[{out_idx}]")
            plt.grid(True)
            plt.show()

    else:
        xs = np.linspace(x_range[0], x_range[1], num_points)
        ys = np.linspace(y_range[0], y_range[1], num_points)
        X, Y = np.meshgrid(xs, ys)
        for out_idx in range(output_dim):
            Z = np.zeros_like(X)
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    outputs = predict_with_normalization(
                        neural_net, normalizer, [X[i, j], Y[i, j]]
                    )
                    Z[i, j] = outputs[out_idx]
            plt.figure()
            plt.pcolormesh(X, Y, Z, shading="auto")
            plt.title(f"Fully connected NN - Output Neuron {out_idx}")
            plt.xlabel("Input x1")
            plt.ylabel("Input x2")
            plt.colorbar(label=f"Output y[{out_idx}]")
            plt.show()
