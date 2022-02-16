import random
import math
from matrix_lib import Matrix

def sigmoid(x):
    return 1 / (1 + math.e ** (-x))

def dsigmoid(y):
    # Why using y is because y has been sigmoid-ed before
    return y * (1-y) 

class NeuralNetwork():
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Persiapan weight antara tiap layer
        self.weight_ih = Matrix(self.hidden_nodes, self.input_nodes)
        self.weight_ho = Matrix(self.output_nodes, self.hidden_nodes)
        ## Inisialisasi weight
        self.weight_ih.randomize()
        self.weight_ho.randomize()

        # Persiapaan bias antara tiap layer
        self.bias_h = Matrix(self.hidden_nodes, 1)
        self.bias_o = Matrix(self.output_nodes, 1)
        # Inisialisasi bias
        self.bias_h.randomize()
        self.bias_o.randomize()

        # Learning rate
        self.learning_rate = 0.1

    def feedforward(self, input_array):

        input = Matrix.fromArray(input_array)

        # Hidden layer
        hidden = Matrix.multiply(self.weight_ih, input)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        # Output Layer
        output = Matrix.multiply(self.weight_ho, hidden)
        output.add(self.bias_o)
        output.map(sigmoid)

        return output.toArray()

    def train(self, input_array, target_array):

        input = Matrix.fromArray(input_array)

        # Hidden layer (Generating hidden outputs)
        hidden = Matrix.multiply(self.weight_ih, input)
        hidden.add(self.bias_h)
        hidden.map(sigmoid)

        # Output layer (Generating output outputs)
        outputs = Matrix.multiply(self.weight_ho, hidden)
        outputs.add(self.bias_o)
        outputs.map(sigmoid)
        
        # Convert targets from array to Matrix
        targets = Matrix.fromArray(target_array)

        # Calculating output error
        output_errors = Matrix.subtract(targets, outputs)
        ## Gradient 
        gradients = Matrix.map_static(outputs, dsigmoid)
        gradients.multiplication(output_errors)
        gradients.multiplication(self.learning_rate)
        ### Calculate Deltas weight between hidden and output
        hidden_transposed = Matrix.transpose(hidden)
        weight_ho_deltas = Matrix.multiply(gradients, hidden_transposed)
        #### Changing the hidden output delta
        self.weight_ho.add(weight_ho_deltas)
        ##### Changing the bias by its deltas (using gradient)
        self.bias_o.add(gradients)

        # Calculating hidden error 
        weight_ho_transposed = Matrix.transpose(self.weight_ho)
        hidden_errors = Matrix.multiply(weight_ho_transposed, output_errors)
        ## Hidden gradient
        hidden_gradients = Matrix.map_static(hidden, dsigmoid)
        hidden_gradients.multiplication(hidden_errors)
        hidden_gradients.multiplication(self.learning_rate)
        ### Calculate delta weight between input and hidden
        input_transposed = Matrix.transpose(input)
        weight_ih_deltas = Matrix.multiply(hidden_gradients, input_transposed)
        #### Changing the input hidden delta
        self.weight_ih.add(weight_ih_deltas)
        ##### Changing the bias by its deltas (using gradient)
        self.bias_h.add(hidden_gradients)

def XOR_testing(number_epochs):
    training_data = [
        ([1,0], [1]),
        ([0,1], [1]),
        ([0,0], [0]),
        ([1,1], [0])
    ]
    neuralnet = NeuralNetwork(2,2,1)

    for i in range(number_epochs):
        random.shuffle(training_data)
        for data in training_data:
            neuralnet.train(data[0], data[-1])
    
    print("Number of Epoch: {}".format(number_epochs))
    print("Weight Input-Hidden: {}".format(neuralnet.weight_ih.data))
    print("Bias Hidden: {}".format(neuralnet.bias_h.data))
    print("Weight Hidden-Output: {}".format(neuralnet.weight_ho.data))
    print("Bias Output: {}".format(neuralnet.bias_o.data))
    print("Input: {}, Predicted Output: {}".format([1,0],neuralnet.feedforward([1,0])))
    print("Input: {}, Predicted Output: {}".format([0,1],neuralnet.feedforward([0,1])))
    print("Input: {}, Predicted Output: {}".format([0,0],neuralnet.feedforward([0,0])))
    print("Input: {}, Predicted Output: {}".format([1,1],neuralnet.feedforward([1,1])))

XOR_testing(5000)