from eightqueens_extended import EightQueensRunner
from utils import Timer
import pandas as pd

if __name__ == '__main__':
    n_range = range(4, 13)
    data = []
    for N in n_range:
        print("\n***********************")
        print(f"board_size={N}")
        size = 5 ** 2 * N  # crossover(top_individuals) * board_size
        times = []
        for _ in range(200):
            with Timer() as t:
                EightQueensRunner(top_individuals=5,
                                  crossover='product',
                                  mutation_rate=1 / 8,
                                  board_size=N,
                                  verbose=False).run()
            times.append(t.getAbsoluteInterval())
        avg_time = sum(times) / len(times)
        data.append({'size': size, 'time': avg_time})
    data = pd.DataFrame(data)
    data.to_csv('time_complexity.csv', index=False)
