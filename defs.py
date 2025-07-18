class Node :
    def __init__ (self) :
        self.value = 0.0

    def return_value (self) :
        return self.value

class Neuron (Node):
    def __init__(self):
        super().__init__()

        self.bias = 0.0
        self.inputs = []

        self.activation = None

        self.output = None

    def add_input(self, input_node):
        self.inputs.append((input_node, 0.0))

    def add_output(self, output_neuron):
        self.output = output_neuron

    def set_activation(self, activation):
        self.activation = activation

class Source (Node):
    def __init__(self):
        super().__init__()

    def set_value (self, value):
        self.value = value


class NeuralLayer:
    def __init__(self, net_size : int) :
        self.net_size = net_size

        self.neurons = [Neuron() for _ in range(self.net_size)]

class NeuralNetwork:
    def __init__(self) :
        self.layers = []
        self.sources = []

    def set_input(self, sources) :
        self.sources = sources

    def add_layer(self, layer : NeuralLayer) :
        self.layers.append(layer)