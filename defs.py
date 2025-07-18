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

    def add_input(self, input_node):
        self.inputs_weights.append((input_node, 0.0))

    def set_activation(self, activation):
        self.activation = activation

    def calculate_output (self) :
        weighted_sum = 0.0

        for input, weight in self.inputs_weights :
            weighted_sum += input.get_value() * weight

        self.value = self.activation(weighted_sum)

    def reset (self) :
        self.inputs_weights = []
        self.bias = 0.0
        self.activation = None

class Source (Node):
    def __init__(self):
        super().__init__()

    def set_value (self, value):
        self.value = value


class NeuralLayer ():
    def __init__(self, layer_size: int, activation):
        self.layer_size = layer_size

        self.neurons = [Neuron() for _ in range(self.layer_size)]

        for neuron in self.neurons :
            neuron.set_activation(activation)

    def calculate_output (self) :
        for neuron in self.neurons :
            neuron.calculate_output()

    def reset(self) :
        for neuron in self.neurons :
            neuron.reset()

class SourceLayer ():
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
        self.layers.append(NeuralLayer(net_size, activation))

    def calculate_output (self) :
        for layer in self.layers :
            layer.calculate_output()

    def get_output (self) :
        vec = []

        for neuron in self.layers[-1].neurons :
            vec.append(neuron.get_value())

    def reset(self):
        for layer in self.layers :
            layer.reset()