import itertools
import math
import random
from utils import Timer
import numpy as np


class Individual:
    """This class represents a 8x8 board in the eight queens puzzle"""

    def __init__(self, state=None):
        """
        :param state: pass in a numpy array of integers to set the state, otherwise will be generated randomly
        """
        self.state = state if state is not None else np.random.randint(0, high=8, size=8)
        self.n = 8

    @staticmethod
    def copy_replace(state: np.ndarray, i: int, x: int) -> np.ndarray:
        """This creates a copy of the state (important as numpy arrays are mutable) with column i set to x"""
        new_state = state.copy()
        new_state[i] = x
        return new_state

    @staticmethod
    def range_missing(start: int, stop: int, missing: int) -> list:
        """
        This creates a list of numbers with a single value missing
        e.g. range_missing(0, 8, 2) -> [0, 1, 3, 4, 5, 6, 7]
        """
        return list(range(start, missing)) + list(range(missing + 1, stop))

    def cost(self) -> int:
        """Calculates the number of pairs attacking"""
        count = 0
        for i in range(len(self.state) - 1):
            # for each queen, look in columns to the right
            # add one to the count if there is another queen in the same row
            count += (self.state[i] == np.array(self.state[i + 1:])).sum()

            # add one to the count for each queen on the upper or lower diagonal
            upper_diagonal = self.state[i] + np.arange(1, self.n - i)
            lower_diagonal = self.state[i] - np.arange(1, self.n - i)
            count += (np.array(self.state[i + 1:]) == upper_diagonal).sum()
            count += (np.array(self.state[i + 1:]) == lower_diagonal).sum()
        return count

    def neighbourhood(self) -> list:
        """This generates every state possible by changing a single queen position"""
        neighbourhood = []
        for column in range(self.n):
            for new_position in self.range_missing(0, self.n, self.state[column]):
                new_state = self.copy_replace(self.state, column, new_position)
                neighbourhood.append(Individual(new_state))

        return neighbourhood

    def random_neighbour(self):
        """Generates a single random neighbour state, useful for some algorithms"""
        column = np.random.choice(range(self.n))
        new_position = np.random.choice(self.range_missing(0, self.n, self.state[column]))
        new_state = self.copy_replace(self.state, column, new_position)
        return Individual(new_state)

    def is_goal(self) -> bool:
        return self.cost() == 0

    def __str__(self):
        if self.is_goal():
            return f"Goal state! {self.state}"
        else:
            return f"{self.state} cost {self.cost()}"


class Population:
    """This class represents a population of individual states"""

    def __init__(self, k: int, crossover_func: staticmethod, states=None):
        self.k = k
        self.crossover_func = crossover_func
        self.population_size = self.crossover_func(k)
        if states is None: self.population = [Individual() for _ in range(self.population_size)]  # initial state
        else: self.population = [Individual(state=states[s]) for s in range(self.population_size)]  # not initial state

    def fitness(self) -> np.ndarray: return np.array([i.cost() for i in self.population])

    def has_the_chad(self) -> bool: return 0 in self.fitness()

    def the_chad(self) -> Individual:
        for individual in self.population:
            if individual.is_goal(): return individual

    def top_k(self) -> np.ndarray:
        fitness = self.fitness()
        return np.argpartition(fitness, self.k)[:self.k]


class EightQueensRunner:
    def __init__(self, k=3, crossover='product', max_iter=1000, n=8, verbose=False):
        self.k = k
        self.crossover = crossover
        self.max_iter = max_iter
        self.n = n
        self.verbose = verbose
        self.selection_split = math.ceil(self.n / 2 - 1)
        # NOTE PERMUTATIONS MUST HAVE k > 2
        self.crossover_func = {
                'permutations': lambda k: math.factorial(k) // math.factorial(k - 2),
                'product': lambda k: k ** 2,
                'combinations_with_rep': lambda k: math.factorial(k + 1) // (2 * math.factorial(k - 1))
        }[crossover]

    @staticmethod
    def permutations(genes_x: list, genes_y: list) -> np.ndarray:
        shuffled = []
        for xi, X in enumerate(genes_x):
            for yi, Y in enumerate(genes_y):
                if xi == yi: continue
                shuffled.append(np.concatenate((X, Y)))
        return shuffled

    @staticmethod
    def product(genes_x: list, genes_y: list) -> np.ndarray:
        return np.array([np.concatenate((X, Y)) for X, Y in itertools.product(genes_x, genes_y)])

    @staticmethod
    def combinations_with_replacement(genes_x: list, genes_y: list) -> np.ndarray:
        shuffled = []
        combinations = set()
        for xi, X in enumerate(genes_x):
            for yi, Y in enumerate(genes_y):
                if f"{xi}{yi}" in combinations: continue
                combinations.update([f"{xi}{yi}", f"{yi}{xi}"])
                shuffled.append(np.concatenate((X, Y)))
        return shuffled

    def run(self):
        generation = 1
        next_generation = None
        while generation < self.max_iter:
            # create new generation
            population = Population(k=self.k, crossover_func=self.crossover_func, states=next_generation)

            # check for the success
            if population.has_the_chad():
                print(f"Goal state achieved in {generation} iterations. {population.the_chad()}")
                break

            # get best individuals
            top_k = [population.population[i] for i in population.top_k()]
            if self.verbose: print(f"Generation {generation} Top {self.k}:", *(str(top).rjust(30) for top in top_k), sep='\n')

            # select genes
            genes_x = []
            genes_y = []
            for top in top_k:
                genes_x.append(top.state[:self.selection_split])
                genes_y.append(top.state[self.selection_split:])
                if self.verbose: print(f"Gene splits: {[top.state[:self.selection_split], top.state[self.selection_split:]]}")

            # crossover
            next_generation = []
            # ab, ac, ba, bc, ca, cb
            if self.crossover == 'permutations': next_generation = self.permutations(genes_x, genes_y)
            # aa, ab, ac, ba, bb, bc, ca, cb, cc
            elif self.crossover == 'product': next_generation = self.product(genes_x, genes_y)
            # aa, ab, ac, bb, bc, cc
            elif self.crossover == 'combinations_with_rep': next_generation = self.combinations_with_replacement(genes_x, genes_y)

            # mutate
            for individual in next_generation:
                # 7/8 chance that individual's single gene will be mutated
                if random.randint(1, 8) == 1: continue
                index = random.randint(0, self.n - 1)
                value = random.randint(0, self.n - 1)
                individual[index] = value

            if self.verbose: print(f"Next generation: {next_generation}\n")
            generation += 1
        else:
            print('The algorithm failed to reach the goal.')


if __name__ == '__main__':
    with Timer():
        p1 = EightQueensRunner(crossover='combinations_with_rep',
                               max_iter=1000,
                               k=3,
                               verbose=False)
        p1.run()
