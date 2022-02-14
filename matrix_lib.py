import random

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
                val = self.data[i][j] # Important thing to note that by doing this, you won't change anything with the matrix itself
                self.data[i][j] = func(val)

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