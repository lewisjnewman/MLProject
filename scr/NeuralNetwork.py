import numpy as np
import json

# The sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

# d/dx(sigmoid function)
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

def tanh(x):
    return np.tanh(x)

# The softmax function
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def rectified_linear(x):
    return np.maximum(0, x)

class NeuralNetwork(object):
    """Class to Represent a Neural Network"""

    # Class for hidden layers
    class Layer(object):
        """Nested Class to represent a layer within the neural network"""

        def __init__(self, input_size, neuron_count, output=False):
            """Init method builds a neural network layer from an input size and a neuron count"""

            self.input_size = input_size
            self.neuron_count = neuron_count
            self.weights = (np.random.rand(input_size, neuron_count)-0.5)*5
            self.biases = np.zeros((1, neuron_count))
            self.output = output


        def forward(self, inputs):
            inputs = np.array(inputs)
            """Runs one layer forward within the neural network"""
            if self.output is False:
                # This is a hidden layer, use sigmoid as the activation function
                return sigmoid(np.dot(inputs, self.weights) + self.biases)
            else:
                # This is a output layer, use softmax as the activation function
                return softmax(np.dot(inputs, self.weights) + self.biases)

        def export_dict(self):
            """Exports the instance as a python dict"""
            return {"weights": self.weights.tolist(), "biases": self.biases.tolist(), "output": self.output, "input_size": self.input_size, "neuron_count": self.neuron_count}

        @staticmethod
        def import_dict(data):
            """Creates a layer from a dict created by the export_dict method"""

            layer = NeuralNetwork.Layer(data["input_size"], data["neuron_count"], data["output"])

            layer.weights = np.array(data["weights"])
            layer.biases = np.array(data["biases"])

            return layer

        def get_weights(self):
            return self.weights

        def get_biases(self):
            return self.biases

        def set_weights(self, weights):
            self.weights = weights

        def set_biases(self, biases):
            self.biases = biases


    def __init__(self, shape):
        assert len(shape) > 1

        self.layers = []
        self.shape = shape

        # Build the hidden layers of the neural network
        for i in range(len(shape)-2):
            self.layers.append(NeuralNetwork.Layer(shape[i], shape[i+1]))

        # Build the final output layer of the neural network
        self.layers.append(NeuralNetwork.Layer(shape[-2], shape[-1], output=True))

    def forward(self, inputs):
        #data = minmax(np.array(inputs))
        data = inputs
        for l in self.layers:
            #print(f"Updated Data {data.tolist()}")
            data = l.forward(data)
        return data

    def export_dict(self):
        return {
            "layers" : list(map(NeuralNetwork.Layer.export_dict,self.layers)),
            "shape": self.shape
        }

    @staticmethod
    def import_dict(data):
        nn = NeuralNetwork(data["shape"])

        nn.layers = []

        for l in data["layers"]:
            nn.layers.append(NeuralNetwork.Layer.import_dict(l))

        return nn

    def save_json(self, filename):
        with open(filename, "w") as fp:
            json.dump(self.export_dict(), fp)


    @staticmethod
    def load_json(filename):
        with open(filename, "r") as fp:
            data = json.load(fp)

        return NeuralNetwork.import_dict(data)

    def GetChromosomes(self):
        """
        Helper function that converts the weights and biases of a neural network into
        a flat array to be used in the genetic algorithm
        """

        chromosomes = []

        for l in self.layers:
            chromosomes += l.get_weights().flatten().tolist() + l.get_biases().flatten().tolist()

        return np.array(chromosomes)

    def SetFromChromosomes(self, chromosomes):
        """
        Helper Function that takes a flat array of floats and turns them into
        the weights and biases of this neural network
        """

        chromosome_indexer = 0

        for i in range(len(self.layers)):
            num_weights = self.layers[i].input_size * self.layers[i].neuron_count
            num_biases = self.layers[i].neuron_count

            weights = chromosomes[chromosome_indexer:(chromosome_indexer+num_weights)]

            chromosome_indexer += num_weights

            biases = chromosomes[chromosome_indexer:(chromosome_indexer+num_biases)]

            chromosome_indexer += num_biases

            self.layers[i].weights = np.array(weights).reshape(self.layers[i].input_size, self.layers[i].neuron_count)
            self.layers[i].biases = np.array(biases).reshape(1, self.layers[i].neuron_count)

# Stuff to train the neural network with a genetic algorithm
def mutate_adjust_chromosomes(chromosomes, probability, mutate_limit):
    """Given a set of chromosomes, a probability and a range, this algorithm
    mutates the chromosomes by some amount"""

    for i in range(len(chromosomes)):
        # Uniform random number to decide whether to mutate
        v = np.random.uniform()

        if v < probability:
            # Do Mutation
            chromosomes[i] += np.random.uniform(-mutate_limit, mutate_limit)

    return chromosomes

def mutate_replace_chromosomes(chromosomes, probability, mutate_limit):
    """Given a set of chromosomes, a probability and a range, this algorithm
    mutates the chromosomes by some amount"""

    for i in range(len(chromosomes)):
        # Uniform random number to decide whether to mutate
        v = np.random.uniform()

        if v < probability:
            # Do Mutation
            chromosomes[i] = np.random.uniform(-mutate_limit, mutate_limit)

    return chromosomes

def merge_chromosomes(c1, c2):
    """Creates an array of chromosomes from the averages of 2 given chromosome arrays"""
    assert len(c1) == len(c2)

    c3 = []
    for i in range(len(c1)):
        c3.append((c1[i] + c2[i]) / 2)

    return c3


def cross_chromosomes(c1, c2):
    """Creates an array of chromosomes using half of 1 array and half of another array"""
    assert len(c1) == len(c2)

    c3 = []
    for i in range(0, len(c1), 2):
        c3.append(c1[i])
        c3.append(c2[i+1])

    return c3
