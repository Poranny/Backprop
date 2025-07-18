class Node :
    def __init__ (self) :
        self.value = 0.0

    def get_value (self) :
        return self.value

class Neuron (Node):
    def __init__(self):
        super().__init__()

        self.bias = 0.0
        self.inputs_weights = []

        self.activation = None

        self.output = None

    def add_input(self, input_node):
        self.inputs_weights.append((input_node, 0.0))

    def add_output(self, output_neuron):
        self.output = output_neuron

    def set_activation(self, activation):
        self.activation = activation

    def calculate_output (self) :
        weighted_sum = 0.0

        for input, weight in self.inputs_weights :
            weighted_sum += input.get_value() * weight

        self.value = self.activation(weighted_sum)

class Source (Node):
    def __init__(self):
        super().__init__()

    def set_value (self, value):
        self.value = value


class NeuralLayer:
    def __init__(self, net_size : int) :
        self.net_size = net_size

        self.neurons = [Neuron() for _ in range(self.net_size)]

    def calculate_output (self) :
        for neuron in self.neurons :
            neuron.calculate_output()

class NeuralNetwork:
    def __init__(self) :
        self.layers = []
        self.sources = []

    def set_input(self, sources) :
        self.sources = sources

    def add_layer(self, layer : NeuralLayer) :
        self.layers.append(layer)

    def calculate_output (self) :
        for layer in self.layers :
            layer.calculate_output()

    def get_output (self) :
        vec = []

        for neuron in self.layers[-1].neurons :
            vec.append(neuron.get_value())