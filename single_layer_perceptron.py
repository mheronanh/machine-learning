def perceptron(p, w, b):
    a = b
    for i in range(len(w)):
        a = a + w[i] * p[i]
    return 1.0 if a >= 0.0 else 0.0

def training(train, number_epoch):
    # Inisialisasi
    weight = [0, 0]
    bias = 0

    for i in range(number_epoch):
        for data in train:
            prediction = perceptron(data, weight, bias)
            error = data[-1] - prediction
            bias = bias + error
            for j in range(len(weight)):
                weight[j] = weight[j] + error * data[j]
        print("Epoch: {}, Weight: {}, Bias: {}".format(i, weight, bias))

    return weight, bias

def evaluate(dataset, w, b):
    for data in dataset:
        prediction = perceptron(data, w, b)
        error = prediction - data[-1]
        if error == 0.0:
            print("Prediction is Correct")
        else:
            print("Prediction is False")

# Dataset
dataset = [[0,0,0], [0,1,0], [1,0,0], [1,1,1]]

# Evaluate over dataset
w, b = training(dataset, 6)
evaluate(dataset, w, b)
