import pandas as pd
import numpy as np
import itertools
from multiprocessing import Pool
from utils import Timer
from eightqueens_extended import EightQueensRunner
import math

cf = {
        'permutations': lambda k: math.factorial(k) // math.factorial(k - 2),
        'product': lambda k: k ** 2,
        'combinations_with_rep': lambda k: math.factorial(k + 1) // (2 * math.factorial(k - 1))
}


def process(params: tuple) -> dict:
    N, K, M, C = params
    size = cf[C](K) * N  # crossover(top_individuals) * board_size
    times = []
    for _ in range(100):
        with Timer(verbose=False) as t:
            EightQueensRunner(top_individuals=K,
                              crossover=C,
                              mutation_rate=M,
                              board_size=N,
                              max_iter=2500).run()
        times.append(t.getAbsoluteInterval())
    avg_time = sum(times) / len(times)
    return {'size': size,
            'time': avg_time,
            'top_individuals': K,
            'crossover': C,
            'mutation_rate': round(M, 6),
            'board_size': N}


if __name__ == '__main__':
    n_range = range(4, 13)
    k_range = range(3, 8)
    m_range = np.linspace(0.05, 0.22, 18)
    c_range = ['product', 'combinations_with_rep', 'permutations']
    data = []

    with Pool(6) as pool:
        for result in pool.imap_unordered(process, itertools.product(n_range, k_range, m_range, c_range), chunksize=2):
            print(result)
            data.append(result)

    data = pd.DataFrame(data)
    data.to_csv('gs_FullSearch.csv', index=False)
