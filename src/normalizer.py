import numpy as np


class Normalizer:
    def __init__(self, data):
        inputs = np.array([row[0] for row in data])
        outputs = np.array([row[1] for row in data])

        self.inputs_mean = np.mean(inputs, axis=0)
        self.inputs_std = np.std(inputs, axis=0)

        self.outputs_mean = np.mean(outputs, axis=0)
        self.outputs_std = np.std(outputs, axis=0)

    def transform(self, data):
        transformed = []
        for inp, out in data:
            inp_norm = ((np.array(inp) - self.inputs_mean) / self.inputs_std).tolist()
            out_norm = ((np.array(out) - self.outputs_mean) / self.outputs_std).tolist()
            transformed.append((inp_norm, out_norm))
        return transformed

    def inverse_transform(self, data):
        inversed = []
        for inp, out in data:
            inp_orig = (np.array(inp) * self.inputs_std + self.inputs_mean).tolist()
            out_orig = (np.array(out) * self.outputs_std + self.outputs_mean).tolist()
            inversed.append((inp_orig, out_orig))
        return inversed

    def transform_inputs(self, data):
        return ((np.array(data) - self.inputs_mean) / self.inputs_std).tolist()

    def inverse_transform_inputs(self, data):
        return (np.array(data) * self.inputs_std + self.inputs_mean).tolist()

    def transform_outputs(self, data):
        return ((np.array(data) - self.outputs_mean) / self.outputs_std).tolist()

    def inverse_transform_outputs(self, data):
        return (np.array(data) * self.outputs_std + self.outputs_mean).tolist()

    def transform_input_number(self, number):
        return (number - self.inputs_mean) / self.inputs_std

    def inverse_transform_input_number(self, number):
        return number * self.inputs_std + self.inputs_mean

    def transform_output_number(self, number):
        return float((number - self.outputs_mean) / self.outputs_std)

    def inverse_transform_output_number(self, number):
        return float(number * self.outputs_std + self.outputs_mean)
