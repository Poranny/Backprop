import random
from typing import List, Tuple
import numpy as np


class Node:
    def __init__(self):
        self.input_connections: List[NeuralConnection] = []
        self.output_connections: List[NeuralConnection] = []

        self.value = 0.0

    def get_value(self):
        return self.value

    def add_input(self, neural_connection):
        self.input_connections.append(neural_connection)

    def add_output(self, neural_connection):
        self.output_connections.append(neural_connection)


class Neuron(Node):
    def __init__(self):
        super().__init__()

        self.bias = random.uniform(-1.0, 1.0)

        self.delta = 0.0

        self.z = 0.0

        self.activation = None

    def set_activation(self, activation):
        self.activation = activation

    def calculate_output(self):
        weights = np.array([conn.weight for conn in self.input_connections])
        values = np.array(
            [conn.input_node.get_value() for conn in self.input_connections]
        )

        weighted_sum = np.sum(values * weights)

        self.z = weighted_sum + self.bias
        self.value = self.activation(self.z)

    def reset(self):
        self.input_connections = []
        self.bias = 0.0
        self.activation = None

    def update_weights(self, learning_rate):
        inputs = np.array(
            [conn.input_node.get_value() for conn in self.input_connections]
        )
        weights = np.array([conn.weight for conn in self.input_connections])

        gradients = self.delta * inputs
        new_weights = weights - learning_rate * gradients

        for conn, w in zip(self.input_connections, new_weights):
            conn.weight = w

        self.bias -= learning_rate * self.delta


class Source(Node):
    def __init__(self):
        super().__init__()

    def set_value(self, value):
        self.value = value


class NeuralConnection:
    def __init__(self, input_node, output_node):
        self.input_node: Node = input_node
        self.output_node: Neuron = output_node
        self.weight: float = random.uniform(-1.0, 1.0)

    def set_weight(self, weight):
        self.weight = weight


class NeuralLayer:
    def __init__(self, layer_size: int, activation, previous_nodes: list[Node]):
        self.layer_size = layer_size

        self.neurons = [Neuron() for _ in range(self.layer_size)]
        self.activation = activation

        for neuron in self.neurons:
            neuron.set_activation(activation)

            for prev_node in previous_nodes:
                new_connection = NeuralConnection(prev_node, neuron)

                neuron.add_input(new_connection)
                prev_node.add_output(new_connection)

    def calculate_output(self):
        for neuron in self.neurons:
            neuron.calculate_output()

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()

    def backprop(self, learning_rate, expected_outputs=None):
        z_vals = np.array([neuron.z for neuron in self.neurons])

        if expected_outputs is not None:
            outputs = np.array([n.get_value() for n in self.neurons])
            errors = outputs - expected_outputs
            activation_derivs = np.array(
                [
                    n.activation(z, is_derivative=True)
                    for n, z in zip(self.neurons, z_vals)
                ]
            )
            deltas = errors * activation_derivs

            for neuron, delta in zip(self.neurons, deltas):
                neuron.delta = delta
        else:
            for neuron in self.neurons:
                sum_deltas = sum(
                    conn.output_node.delta * conn.weight
                    for conn in neuron.output_connections
                )
                neuron.delta = (
                    neuron.activation(neuron.z, is_derivative=True) * sum_deltas
                )

        for neuron in self.neurons:
            neuron.update_weights(learning_rate)

    def get_weights(self):
        all_weights = []
        all_biases = []

        for neuron in self.neurons:
            weights = [conn.weight for conn in neuron.input_connections]
            all_weights.append(weights)
            all_biases.append(neuron.bias)

        return (all_weights, all_biases), self.activation


class SourceLayer:
    def __init__(self, layer_size: int):
        self.layer_size = layer_size

        self.sources = [Source() for _ in range(self.layer_size)]

    def set_values(self, values):
        if len(values) != self.layer_size:
            raise ValueError(
                f"Expected {self.layer_size} values, but got {len(values)}"
            )

        for i in range(self.layer_size):
            self.sources[i].set_value(values[i])


class NeuralNetwork:
    def __init__(self):
        self.sourceLayer = None
        self.layers = []

    def create_source_layer(self, source_size):
        self.sourceLayer = SourceLayer(source_size)

    def set_source_inputs(self, sources):
        self.sourceLayer.set_values(sources)

    def add_layer(self, net_size: int, activation: callable):
        if self.sourceLayer is None:
            raise RuntimeError("Source layer not initialized")

        if len(self.layers) == 0:
            self.layers.append(
                NeuralLayer(net_size, activation, self.sourceLayer.sources)
            )
        else:
            self.layers.append(
                NeuralLayer(net_size, activation, self.layers[-1].neurons)
            )

    def calculate_output(self):
        for layer in self.layers:
            layer.calculate_output()

    def get_output(self):
        vec = []

        for neuron in self.layers[-1].neurons:
            vec.append(neuron.get_value())

        return vec

    def backprop(self, expected_output, learning_rate):
        if len(self.layers) == 0:
            raise RuntimeError("No neural network to backprop")

        self.layers[-1].backprop(learning_rate, expected_output)

        for layer in reversed(self.layers[0:-1]):
            layer.backprop(learning_rate)

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def get_weights(self):
        all_layers_weights: List[Tuple[List[Tuple[List[float], float]], callable]] = []
        for layer in self.layers:
            all_layers_weights.append(layer.get_weights())
        return all_layers_weights
