class Neuron:
    def __init__(self) :
        self.bias = 0.0
        self.inputs = []

        self.activation = None

        self.output = None

    def add_input(self, input_neuron):
        self.inputs.append((input_neuron, 0.0))

    def add_output(self, output_neuron):
        self.output = output_neuron

    def set_activation(self, activation):
        self.activation = activation

class Layer:
    def __init__(self, net_size : int) :
        self.net_size = net_size

        self.neurons = [Neuron() for _ in range(self.net_size)]