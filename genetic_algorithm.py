import numpy as np

class GeneticAlgorithm():
    def __init__(self, objective, bounds, n_bits, n_iter, n_pop, r_cross):
        # Parameter
        self.objective = objective
        self.bounds = bounds
        self.n_pop = n_pop
        self.n_bits = n_bits
        self.n_iter = n_iter
        self.r_cross = r_cross
        self.r_mut = 1.0 / (float(n_bits) * len(bounds))
        self.population = []
        for _ in range(self.n_pop):
            chromosome = np.random.randint(0, 2, size = n_bits*len(self.bounds)).tolist()
            self.population.append(chromosome)
        self.scores = []

        # Result from selected
        self.selected = []

        # Result from crossover
        self.crossovered = []

        # Initiate final value
        self.best = 0
        self.best_eval = self.objective(self.decode(self.bounds, self.n_bits, self.population[0]))
    
    def selection(self, k=3):
        # Using Tournament Selection with k=3 as default parameter
        for _ in range(self.n_pop):
            selection_index = np.random.randint(len(self.population))
            for index in np.random.randint(0, len(self.population), k-1):
                if self.scores[index] < self.scores[selection_index]:
                    selection_index = index
            self.selected.append(self.population[selection_index])

    def mutation(self):
        for c in self.crossovered:       
            for i in range(len(c)):
                if np.random.rand() < self.r_mut:
                    c[i] = 1 - c[i]

    def process(self):
        for generation in range(self.n_iter):
            decoded = [self.decode(self.bounds, self.n_bits, p) for p in self.population]
            self.scores = [self.objective(d) for d in decoded]
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
        
        decoded = self.decode(self.bounds, self.n_bits, self.best)
        print('Final value f(%s) = %f' % (decoded, self.best_eval))

    @staticmethod
    def decode(bounds, n_bits, bitstring):
        decoded = []
        largest = 2**n_bits
        for i in range(len(bounds)):
            start, end = i * n_bits, (i * n_bits)+ n_bits
            substring = bitstring[start:end]
            chars = ''.join([str(s) for s in substring])
            integer = int(chars, 2)
            value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])
            decoded.append(value)
        return decoded

    @staticmethod
    def crossover(p1, p2, r_cross):
        c1, c2 = p1.copy(), p2.copy()
        if np.random.rand() < r_cross:
            cross_point = np.random.randint(1, len(p1)-2)
            c1 = p1[:cross_point] + p2[cross_point:]
            c2 = p2[:cross_point] + p1[cross_point:]
        return [c1, c2]

'''
Testing Number 2
Given equation f(x) = x + 10 sin(2x) with this given bounds: 0 <= x <= 10  
'''
def objective1(x):
	return x[0] + 10 * np.sin(2*x[0])
prob1 = GeneticAlgorithm(objective1, [[0.0, 10.0]], 16, 100, 100, 0.9)
print("Problem set 1: ")
prob1.process()

'''
Testing Number 2
Given equation f(x, y) = x^2 + y^2 with this given bounds: -5 <= x, y <= 5  
'''
def objective2(x):
	return x[0]**2.0 + x[1]**2.0
prob2 = GeneticAlgorithm(objective2, [[-5.0, 5.0], [-5.0, 5.0]], 16, 100, 100, 0.9)
print("Problem set 2: ")
prob2.process()





     

            

        


