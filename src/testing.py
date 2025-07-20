def predict_with_normalization(neural_net, normalizer, inputs):
    if normalizer is None:
        neural_net.set_source_inputs(inputs)
        neural_net.calculate_output()
        return neural_net.get_output()

    inputs_norm = normalizer.transform_inputs(inputs)

    neural_net.set_source_inputs(inputs_norm)
    neural_net.calculate_output()

    outputs = neural_net.get_output()
    outputs_norm = normalizer.inverse_transform_outputs(outputs)

    return outputs_norm
