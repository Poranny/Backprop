import pytest

from src.activation_functions import Identity
from src.neural_defs import (
    SourceLayer,
    NeuralConnection,
    NeuralNetwork,
    Neuron,
    Source,
)


def test_source_layer_set_values_success_and_failure():
    sl = SourceLayer(3)

    sl.set_values([0.1, 0.2, 0.3])
    assert [s.get_value() for s in sl.sources] == [0.1, 0.2, 0.3]

    with pytest.raises(ValueError):
        sl.set_values([1.0, 2.0])


def test_neural_connection_and_node_linking():
    src = Source()
    dst = Neuron()
    conn = NeuralConnection(src, dst)

    dst.add_input(conn)
    src.add_output(conn)
    assert conn in src.output_connections
    assert conn in dst.input_connections

    assert hasattr(conn, "weight") and isinstance(conn.weight, (int, float))


def test_single_neuron_forward_pass_and_get_weights():
    src = Source()
    src.set_value(3.0)
    neuron = Neuron()
    neuron.set_activation(Identity)

    conn = NeuralConnection(src, neuron)
    conn.set_weight(2.0)
    neuron.add_input(conn)
    src.add_output(conn)
    neuron.bias = 1.0

    neuron.calculate_output()
    # 2*3 + 1 = 7
    assert neuron.get_value() == pytest.approx(7.0)

    wgts = [c.weight for c in neuron.input_connections]
    biases = [neuron.bias]
    act = neuron.activation

    assert isinstance(wgts[0], (int, float))
    assert isinstance(biases[0], (int, float))
    assert callable(act) and act is Identity


def test_neural_network_forward_and_get_output():

    nn = NeuralNetwork()
    nn.create_source_layer(1)
    nn.set_source_inputs([4.0])
    nn.add_layer(1, Identity)
    nn.add_layer(1, Identity)

    l0 = nn.layers[0].neurons[0]
    c0 = l0.input_connections[0]
    c0.weight = 0.5
    l0.bias = 0.0

    l1 = nn.layers[1].neurons[0]
    c1 = l1.input_connections[0]
    c1.weight = 2.0
    l1.bias = -1.0

    nn.calculate_output()
    out = nn.get_output()
    # 0.5*4+0=2.0, 2.0*2+(-1)=3.0
    assert out == [pytest.approx(3.0)]


def test_neural_network_errors_and_backprop():
    nn = NeuralNetwork()

    with pytest.raises(RuntimeError):
        nn.add_layer(1, Identity)

    with pytest.raises(RuntimeError):
        nn.backprop([1.0], learning_rate=0.1)

    nn.create_source_layer(1)
    nn.set_source_inputs([0.0])
    nn.add_layer(1, Identity)
    nn.calculate_output()

    nn.backprop([0.5], learning_rate=0.1)

    ws = nn.get_weights()
    assert isinstance(ws, list)
