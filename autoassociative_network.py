from matrix_lib import Matrix
import matplotlib.pyplot as plt

def hardlims(x):
    return 1 if x > 0 else -1

class AutoassociativeMemory():
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.size = int(self.height * self.width)
        self.weight = Matrix(self.size, self.size)
        self.weight.randomize()
        self.learning_rate = 0.1

    @staticmethod
    def GridToInput(a):
        result = []
        for i in range(len(a)):
            for j in range(len(a[i])):
                result.append(a[i][j])
        return result

    @staticmethod
    def OutputToGrid(amn, output):
        result = []
        pointer = 0
        for i in range(amn.height):
            val = output[pointer:(pointer+amn.width)]
            result.append(val)
            pointer += amn.width
        return result

    def feedforward(self, input_grid):
        input_arr = AutoassociativeMemory.GridToInput(input_grid)
        input = Matrix.fromArray(input_arr)

        output = Matrix.multiply(self.weight, input)
        output.map(hardlims)

        return output.toArray()

    def train(self, input_grid):
        input_arr = AutoassociativeMemory.GridToInput(input_grid)
        input = Matrix.fromArray(input_arr)
        input_T = Matrix.transpose(input)

        deltaWeight = Matrix.multiply(input, input_T)
        deltaWeight.multiplication(self.learning_rate)
        self.weight.add(deltaWeight)

def drawing(dataset, filename):
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(dataset)
    plt.colorbar(pixel_plot)
    plt.savefig('{}.png'.format(filename))

def digit_recognition(number_epochs):
    zero = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1]
    ]

    one = [
        [-1,1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,-1,-1],
    ]

    two = [
        [1,1,1,-1,-1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,1,-1],
        [-1,1,1,-1,-1],
        [-1,1,-1,-1,-1],
        [-1,1,1,1,1]
    ]

    test_digit = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1]
    ]

    dataset = [zero, one, two]
    amn = AutoassociativeMemory(6,5)

    for i in range(number_epochs):
        for data in dataset:
            amn.train(data)
    
    test = amn.feedforward(test_digit)
    drawing(test_digit, 'Test Data')
    drawing(AutoassociativeMemory.OutputToGrid(amn, test), 'Recognition')

digit_recognition(100)