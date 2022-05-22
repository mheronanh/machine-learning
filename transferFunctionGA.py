import numpy as np
import control as control
import matplotlib.pyplot as plt

class GeneticAlgorithm():
    def __init__(self, n_bits, n_iter, n_pop, n_zero, n_pole, r_cross):
        self.n_pop = n_pop
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.n_zero = n_zero
        self.n_pole = n_pole
        self.r_mut = 1.0 / (float(n_bits) * (n_pole + n_zero))
        self.population = []
        for _ in range(self.n_pop):
            chromosome = np.random.randint(0, 2, size = n_bits*(n_pole + n_zero)).tolist()
            self.population.append(chromosome)
        self.scores = []
        self.selected = []
        self.crossovered = []

        # Initialize data
        self.xData, self.yData = self.transferFunction([1., 1., 1., 1.], [1., 3, 2, 5], 0., 3, 50)

        self.best = 0
        self.best_eval = self.fitnessScoring(self.xData, self.yData, self.decode(self.n_bits, self.n_zero, self.n_pole, self.population[0]), self.n_zero, self.n_pole)

    def selection(self, k=3):
        # Using Tournament Selection with k=3 as default parameter
        for _ in range(self.n_pop):
            bestSelection = max(self.scores)
            selection_index = self.scores.index(bestSelection)
            for index in np.random.randint(0, len(self.population), k-1):
                if self.scores[index] < self.scores[selection_index]:
                    selection_index = index
            self.selected.append(self.population[selection_index])

    def mutation(self):
        # Using flip bit mutation
        for c in self.crossovered:       
            for i in range(len(c)):
                if np.random.rand() < self.r_mut:
                    c[i] = 1 - c[i]
    
    @staticmethod
    def crossover(p1, p2, r_cross):
        # Using 1-point crossover
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            cross_point = np.random.randint(1, len(p1)-2)
            c1 = p1[:cross_point] + p2[cross_point:]
            c2 = p2[:cross_point] + p1[cross_point:]
        return [c1, c2]

    def process(self):
        for generation in range(self.n_iter):
            print("Iteration: {}".format(generation))
            decoded = [self.decode(self.n_bits, self.n_zero, self.n_pole, p) for p in self.population]
            self.scores = [self.fitnessScoring(self.xData, self.yData, d, self.n_zero, self.n_pole) for d in decoded]
            for i in range(self.n_pop):
                if self.scores[i] < self.best_eval:
                    self.best, self.best_eval = self.population[i], self.scores[i]
                    print("Generation: %d, New Best f(%s) = %f" % (generation,  decoded[i], self.scores[i]))
            self.selection()
            children = []
            for i in range(0, self.n_pop, 2):
                p1, p2 = self.selected[i], self.selected[i+1]
                self.crossovered = self.crossover(p1, p2, self.r_cross)
                self.mutation()
                for c in self.crossovered:
                    children.append(c)
            self.population = children
        
        decoded = self.decode(self.n_bits, self.n_zero, self.n_pole, self.best)
        print('Final value f(%s) = %f' % (decoded, self.best_eval))

    @staticmethod
    def transferFunction(numerator, denumerator, startPoint, endPoint, sampleSize):
        t = np.linspace(startPoint, endPoint, sampleSize)
        sys = control.tf(numerator, denumerator)
        T, yOut = control.step_response(sys, T=t)
        return T, yOut

    @staticmethod
    def fitnessScoring(xData, yData, coefTest, n_zero, n_pole):
        sys = control.tf(coefTest[:n_zero], coefTest[n_zero:])
        T, yOut =  control.step_response(sys, T=xData)
        fitnessScore = 0
        for i in range(len(yData)):
            fitnessScore += np.abs(yData[i]-yOut[i])
        return fitnessScore
    
    @staticmethod
    def decode(n_bits, n_zero, n_pole, bitstring):
        decoded = []
        largest = 2**n_bits
        state = 1
        while state <= 2:
            if state == 1:
                n = n_zero
            else:
                n = n_pole
            i = 0
            for i in range(n):
                start, end = i * n_bits, (i * n_bits) + n_bits
                substring = bitstring[start:end]
                chars = ''.join([str(s) for s in substring])
                integer = int(chars, 2)
                value = (integer/largest)
                decoded.append(value)
            state += 1
        return decoded

a = GeneticAlgorithm(8, 100, 1500, 4, 4, 0.9)
a.process()

'''
t = np.linspace(0., 3, 50)
sys_prob = control.tf([1.], [1., 3, 2, 5])
print(sys_prob)
sys_ans = control.tf([0.171875], [0.171875, 0.5625, 0.27734375, 0.93359375])
T, yout_prob = control.impulse_response(sys_prob, T=t)
T, yout_ans = control.impulse_response(sys_ans, T=t)

plt.plot(T, yout_prob, 'r', label = 'Target')
plt.plot(T, yout_ans, 'b', label = 'Answer')
plt.title('System Response')
plt.legend()
plt.show()
'''