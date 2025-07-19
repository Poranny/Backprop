from typing import List


class Node :
    def __init__ (self) :
        self.input_connections : List[NeuralConnection] = []
        self.output_connections : List[NeuralConnection] = []

        self.value = 0.0

    def get_value (self) :
        return self.value

    def add_input(self, neural_connection):
        self.input_connections.append(neural_connection)

    def add_output(self, neural_connection):
        self.output_connections.append(neural_connection)


class Neuron (Node):
    def __init__(self):
        super().__init__()

        self.bias = 0.0

        self.delta = 0.0

        self.activation = None

    def set_activation(self, activation):
        self.activation = activation

    def calculate_output (self) :
        weighted_sum = 0.0

        for connection in self.input_connections :
            weighted_sum += connection.input_neuron.get_value() * connection.weight

        self.value = self.activation(weighted_sum)

    def reset (self) :
        self.input_connections = []
        self.bias = 0.0
        self.activation = None

    def update_weights(self, learning_rate):
        for i, connection in enumerate(self.input_connections):
            gradient = self.delta * connection.input_neuron.get_value()

            new_weight = connection.weight - learning_rate * gradient
            self.input_connections[i].weight = new_weight

        self.bias -= learning_rate * self.delta

class Source (Node):
    def __init__(self):
        super().__init__()

    def set_value (self, value):
        self.value = value

class NeuralConnection :
    def __init__ (self, input_neuron, output_neuron) :
        self.input_neuron : Neuron = input_neuron
        self.output_neuron : Neuron = output_neuron
        self.weight : float = 1.0

    def set_weight(self, weight) :
        self.weight = weight


class NeuralLayer:
    def __init__(self, layer_size: int, activation, previous_nodes : list[Node]) :
        self.layer_size = layer_size

        self.neurons = [Neuron() for _ in range(self.layer_size)]

        for neuron in self.neurons :
            neuron.set_activation(activation)

            for node in previous_nodes :
                new_connection = NeuralConnection(neuron, node)
                neuron.add_input(new_connection)
                node.add_output(new_connection)

    def calculate_output (self) :
        for neuron in self.neurons :
            neuron.calculate_output()

    def reset(self) :
        for neuron in self.neurons :
            neuron.reset()

    def backprop (self, expected_output, learning_rate) :

        for i, neuron in enumerate(self.neurons) :
            error = neuron.get_value() - expected_output[i]

            delta = error * neuron.activation(neuron.get_value())

            neuron.delta = delta

            neuron.update_weights(learning_rate)

class SourceLayer:
    def __init__(self, layer_size: int):
        self.layer_size = layer_size

        self.sources = [Source() for _ in range(self.layer_size)]


    def set_values (self, values) :
        if len(values) != self.layer_size :
            raise ValueError(f"Expected {self.layer_size} values, but got {len(values)}")

        for i in range(self.layer_size) :
            self.sources[i].set_value(values[i])

class NeuralNetwork:
    def __init__(self) :
        self.sourceLayer = None
        self.layers = []

    def create_source_layer(self, source_size) :
        self.sourceLayer = SourceLayer(source_size)

    def set_source_inputs (self, sources) :
        self.sourceLayer.set_values(sources)

    def add_layer(self, net_size : int, activation : callable) :
        if self.sourceLayer is None:
            raise RuntimeError ("Source layer not initialized")

        if len(self.layers) == 0 :
            self.layers.append(NeuralLayer(net_size, activation, self.sourceLayer.sources))
        else :
            self.layers.append (NeuralLayer(net_size, activation, self.layers[-1].neurons))

    def calculate_output (self) :
        for layer in self.layers :
            layer.calculate_output()

    def get_output (self) :
        vec = []

        for neuron in self.layers[-1].neurons :
            vec.append(neuron.get_value())

        return vec

    def backprop(self, expected_output, learning_rate):
        if len(self.layers) == 0 :
            raise RuntimeError ("No neural network to backprop")


        for layer in reversed(self.layers) :
            layer.backprop(expected_output, learning_rate)

    def reset(self):
        for layer in self.layers :
            layer.reset()