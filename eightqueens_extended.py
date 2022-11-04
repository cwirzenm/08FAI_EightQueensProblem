from utils import Timer
import itertools
import math
import random
import numpy as np


class Individual:
    """This class represents an individual and contains his genome"""

    def __init__(self, state=None, board_size=8):
        if state is None:
            self.n = board_size
            self.state = np.random.randint(0, high=board_size, size=board_size)
        else:
            if len(state) != board_size:
                raise Exception('Board size and states are not matching.')
            self.n = board_size
            self.state = state
        self.fitness = None

    def cost(self) -> int:
        """Calculates the number of pairs attacking"""
        if self.fitness is None:
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
            self.fitness = count
        return self.fitness

    def is_goal(self) -> bool:
        return self.cost() == 0

    def __str__(self):
        if self.is_goal(): return f"Goal state! {self.state}"
        else: return f"{self.state} cost {self.cost()}"


class Population:
    """This class represents a population of individual states"""

    def __init__(self, top_individuals: int, board_size: int, crossover: staticmethod, states=None):
        self.k = top_individuals
        self.n = board_size
        self.cf = crossover
        self.population_size = crossover(top_individuals)

        if states is None: self.population = [Individual(board_size=board_size) for _ in range(self.population_size)]  # initial state
        else: self.population = [Individual(state=states[s], board_size=board_size) for s in range(self.population_size)]  # not initial state

    def fitness(self) -> np.ndarray: return np.array([i.cost() for i in self.population])

    def has_the_goal(self) -> bool: return 0 in self.fitness()

    def the_goal(self) -> Individual:
        for individual in self.population:
            if individual.is_goal(): return individual

    def top_k(self) -> np.ndarray:
        fitness = self.fitness()
        return np.argpartition(fitness, self.k)[:self.k]


class EightQueensRunner:
    """This is a controller class for the eight queens problem"""

    def __init__(self, top_individuals: int, crossover: str, mutation_rate: float, board_size: int, max_iter=5000, verbose=False):
        """
        :param top_individuals: the amount of top individuals
        :param crossover: the way in which genes from the best individual crossover
        :param mutation_rate: rate at which the genes mutate
        :param board_size: determines the size of board
        :param max_iter: upper limit for the number of iterations
        :param verbose: verbosity level
        """
        self.k = top_individuals
        self.c = crossover
        self.m = mutation_rate
        self.n = board_size
        self.max_iter = max_iter
        self.verbose = verbose

        if board_size < 4: raise Exception('No solution for board size smaller than 4.')
        if crossover == 'permutations' and top_individuals < 3: raise Exception("Permutations crossover require top_individuals > 2.")
        self.selection_split = math.ceil(self.n / 2 - 1)
        self.cf = {
                'permutations': lambda k: math.factorial(k) // math.factorial(k - 2),
                'product': lambda k: k ** 2,
                'combinations_with_rep': lambda k: math.factorial(k + 1) // (2 * math.factorial(k - 1))
        }[crossover]

    @staticmethod
    def permutations(genes_x: list, genes_y: list) -> np.ndarray:
        """ab, ac, ba, bc, ca, cb"""
        shuffled = []
        for xi, X in enumerate(genes_x):
            for yi, Y in enumerate(genes_y):
                if xi == yi: continue
                shuffled.append(np.concatenate((X, Y)))
        return shuffled

    @staticmethod
    def product(genes_x: list, genes_y: list) -> np.ndarray:
        """aa, ab, ac, ba, bb, bc, ca, cb, cc"""
        return np.array([np.concatenate((X, Y)) for X, Y in itertools.product(genes_x, genes_y)])

    @staticmethod
    def combinations_with_rep(genes_x: list, genes_y: list) -> np.ndarray:
        """aa, ab, ac, bb, bc, cc"""
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
            population = Population(top_individuals=self.k, board_size=self.n, crossover=self.cf, states=next_generation)

            # check for the success
            if population.has_the_goal():
                # print(f"Goal state achieved in {generation} iterations. {population.the_goal()}")
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
            if self.verbose: print(f"Gene splits:", *(str(gene).rjust(23) for gene in np.concatenate((genes_x, genes_y), 1)), sep='\n')

            # crossover
            next_generation = getattr(self, self.c)(genes_x, genes_y)

            # mutate
            for i, individual in enumerate(next_generation):
                for j, gene in enumerate(individual):
                    # m chance that a gene will be mutated
                    if random.uniform(0, 1) >= self.m: continue
                    next_generation[i][j] = random.choice([r for r in range(0, self.n) if r != gene])

            if self.verbose: print(f"Next generation:", *(str(gene).rjust(23) for gene in next_generation), sep='\n', end='\n\n')
            generation += 1
        else:
            # print('The algorithm failed to reach the goal.')
            pass


if __name__ == '__main__':
    with Timer() as t:
        EightQueensRunner(top_individuals=5,
                          crossover='permutations',
                          mutation_rate=1 / 8,
                          board_size=8,
                          verbose=True).run()
    print(t.getAbsoluteInterval())
