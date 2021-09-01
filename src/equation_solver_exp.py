
from typing import List, Tuple, Optional
import time
from logging import basicConfig, getLogger
import logging

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

try:
    from icecream import ic
except ImportError:
    def ic(*args):
        print("ic: " + str(args))
        return args

from abstract_solver import AbstractEquationSolver

class EquationSolverExp(AbstractEquationSolver):
    def _fitnessEquation(self, w: np.ndarray) -> np.ndarray:
        Y_sol = self.calcY(w)
        ret = np.abs(
            np.gradient(Y_sol, self.x, axis=1) - Y_sol
        )
        return ret

    def _fitnessInitialCondition(self, w: np.ndarray) -> np.ndarray:
        ret = np.abs(
            1 - w[:, 0]
        )
        return ret
    
    # @override
    def calc_fitness(self, w: np.ndarray) -> np.ndarray:
        fitEq = self._fitnessEquation(w)
        m = fitEq.shape[1]
        fitEq = np.sqrt(np.power(fitEq, 2).sum(axis=1) / m).reshape(-1)
        fitEq /= fitEq.std()
        fitIni = self._fitnessInitialCondition(w).reshape(-1)
        fitSum = fitEq + fitIni
        return fitSum


def main():
    basicConfig(
        format='\x1b[34m%(levelname)s\x1b[0m: [\x1b[32m%(funcName)s\x1b[0m] \x1b[36m%(message)s\x1b[0m',
        level=logging.INFO
    )
    
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}) #桁を揃える
    logger = getLogger('Exp-Solver')
    
    a = -1
    b = 1
    c = 9
    M = 258
    GENERATION = 2**10
    POPULATION = 258
    beta = 0.1
    mu = 0.05
    
    solver = EquationSolverExp(a, b, c, M, population=POPULATION, beta=beta, mu=mu, logger=logger, seed=64)
    
    st = time.time()
    
    # w_best = solver.solveWithWatching(generation=GENERATION, analytical=np.exp)
    w_best = solver.solve(GENERATION)
    et = time.time()
    
    logger.info(f"finish solving. time: {et - st: .4f} (s)")
    
    y_solve = solver.calcY(w_best)
    
    x = np.linspace(a, b, M)
    y_analytical = np.exp(x)
    
    mse = solver.score(y_solve, y_analytical)
    
    logger.info(f"MSE = {mse: .4e}")
    
    axs: List[Axes]
    fig, axs = plt.subplots(1, 2)
    
    ax0 = axs[0]
    ax0.set_title('Analytical answer and solved y')
    ax0.plot(x, y_analytical, label='Analytical')
    ax0.scatter(x[::6], y_solve[::6], s=12, marker='*', color='green', label='$y_{solve}$')
    ax0.legend()
    
    ax1 = axs[1]
    fitnessHistory = solver.fitnessHistory
    ax1.plot(range(len(fitnessHistory)), fitnessHistory)
    ax1.set_title('Best Fitness')
    
    plt.show()

if __name__ == '__main__':
    main()
