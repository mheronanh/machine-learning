from matrix_lib import Matrix
import matplotlib.pyplot as plt
import random

def hardlims(x):
    return 1 if x > 0 else -1

class AutoassociativeMemory():
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.size = int(self.height * self.width)
        self.weight = Matrix(self.size, self.size)
        self.weight.randomize()
        self.learning_rate = 1

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

def drawing(dataset, pathname, filename):
    pixel_plot = plt.figure()
    pixel_plot = plt.imshow(dataset)
    plt.colorbar(pixel_plot)
    plt.savefig('{}/{}.png'.format(pathname, filename))

def digits():
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

    three = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,1,-1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1]
    ]

    four = [
        [-1,-1,-1,1,-1],
        [-1,-1,1,1,-1],
        [-1,1,-1,1,-1],
        [1,-1,-1,1,-1],
        [1,1,1,1,1],
        [-1,-1,-1,1,-1]
    ]

    five = [
        [1,1,1,1,1],
        [1,-1,-1,-1,-1],
        [1,1,1,1,-1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,1],
        [1,1,1,1,-1]
    ]

    six = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,-1],
        [1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1]
    ]

    seven = [
        [1,1,1,1,-1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,1,-1],
    ]

    eight = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1]
    ]

    nine = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,1],
        [-1,-1,-1,-1,1],
        [-1,1,1,1,-1]
    ]

    zero_50 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    one_50 = [
        [-1,1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    two_50 = [
        [1,1,1,-1,-1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    three_50 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    four_50 = [
        [-1,-1,-1,1,-1],
        [-1,-1,1,1,-1],
        [-1,1,-1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    five_50 = [
        [1,1,1,1,1],
        [1,-1,-1,-1,-1],
        [1,1,1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    six_50 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,-1],
        [1,1,1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    seven_50 = [
        [1,1,1,1,-1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    eight_50 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,1,1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    nine_50 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    zero_67 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    one_67 = [
        [-1,1,1,-1,-1],
        [-1,-1,1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    two_67 = [
        [1,1,1,-1,-1],
        [-1,-1,-1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    three_67 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    four_67 = [
        [-1,-1,-1,1,-1],
        [-1,-1,1,1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    five_67 = [
        [1,1,1,1,1],
        [1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    six_67 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    seven_67 = [
        [1,1,1,1,-1],
        [-1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    eight_67 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    nine_67 = [
        [-1,1,1,1,-1],
        [1,-1,-1,-1,1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
        [-1,-1,-1,-1,-1],
    ]

    dataset = [zero, one, two, three, four, five, six, seven, eight, nine]
    dataset_50 = [zero_50, one_50, two_50, three_50, four_50, five_50, six_50, seven_50, eight_50, nine_50]
    dataset_67 = [zero_67, one_67, two_67, three_67, four_67, five_67, six_67, seven_67, eight_67, nine_67]

    dataset_noisy = []
    for digit in dataset:
        digit_noisy = digit.copy()
        random.shuffle(digit_noisy)
        dataset_noisy.append(digit_noisy)

    return dataset, dataset_50, dataset_67, dataset_noisy

def digit_recognition():

    dataset, dataset_50, dataset_67, dataset_noisy = digits()

    amn = AutoassociativeMemory(6,5)

    for i, data in enumerate(dataset):
        drawing(data, './digit_normal','Digit {}'.format(i))
        amn.train(data)

    for i, data_noisy in enumerate(dataset_noisy):
        drawing(data_noisy, './digit_noisy', 'Noisy Digit of {}'.format(i))

    for i, data_50 in enumerate(dataset_50):
        drawing(data_50, './digit_50', '50% Digit of {}'.format(i))

    for i, data_67 in enumerate(dataset_67):
        drawing(data_67, './digit_67', '67% Digit of {}'.format(i))
    
    for i, test_50 in enumerate(dataset_50):
        test = amn.feedforward(test_50)
        drawing(AutoassociativeMemory.OutputToGrid(amn, test), './digit_50_predicted','Recognition of Digit {}'.format(i))

    for i, test_67 in enumerate(dataset_67):
        test = amn.feedforward(test_67)
        drawing(AutoassociativeMemory.OutputToGrid(amn, test), './digit_67_predicted','Recognition of Digit {}'.format(i))

    for i, test_noisy in enumerate(dataset_noisy):
        test = amn.feedforward(test_noisy)
        drawing(AutoassociativeMemory.OutputToGrid(amn, test), './digit_noisy_predicted','Recognition of Digit {}'.format(i))

digit_recognition()