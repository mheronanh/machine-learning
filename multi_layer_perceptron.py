import random
import math

class Matrix():
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.data = []

        for i in range(self.rows):
            self.data.append([])
            for j in range(self.cols):
                self.data[i].append(0)
    
    def randomize(self):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] += random.randint(-1,1)

    def add(self, n):
        if type(n) == Matrix:
            # Element-wise
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n.data[i][j]   
        else:
            # Scalar
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] += n
    
    def map(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.data[i][j] = func(self.data[i][j])

    def print(self):
        for i in range(self.rows):
            print(self.data[i])

    def toArray(self):
        result = []
        for i in range(self.rows):
            for j in range(self.cols):
                result.append(self.data[i][j])
        return result
    
    def multiplication(self, n):
        # Different with mstatic method multiply, using these for element-wise operation
        if type(n) == Matrix: 
            # Hadamard Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n.data[i][j]
        else: 
            # Scalar Product
            for i in range(self.rows):
                for j in range(self.cols):
                    self.data[i][j] *= n

    @staticmethod
    def map_static(matrix, func):
        result = Matrix(matrix.rows, matrix.cols)
        for i in range(matrix.rows):
            for j in range(matrix.cols):
                result.data[i][j] = func(matrix.data[i][j])
        return result

    @staticmethod
    def transpose(a):
        result = Matrix(a.cols, a.rows)
        for i in range(a.cols):
            for j in range(a.rows):
                result.data[i][j] = a.data[j][i]
        return result

    @staticmethod
    def fromArray(array):
        result = Matrix(len(array), 1)
        for i in range(len(array)):
            result.data[i][0] = array[i]
        return result

    @staticmethod
    def subtract(a, b):
        result = Matrix(a.rows, a.cols)
        for i in range(a.rows):
            for j in range(a.cols):
                result.data[i][j] = a.data[i][j] - b.data[i][j]
        return result

    @staticmethod
    def multiply(a, b):
        if a.cols != b.rows:
            print("Column of A must match rows of B")
            return False
        else:
            result = Matrix(a.rows, b.cols)
            for i in range(result.rows):
                for j in range(result.cols):
                    sum = 0
                    for k in range(a.cols):
                        sum += a.data[i][k] * b.data[k][j]
                    result.data[i][j] = sum
            return result  

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
    
    print(neuralnet.feedforward([1,0]))
    print(neuralnet.feedforward([0,1]))
    print(neuralnet.feedforward([0,0]))
    print(neuralnet.feedforward([1,1]))

XOR_testing(50000)