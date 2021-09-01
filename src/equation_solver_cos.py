
import time
from typing import List, Tuple, Optional
import logging
from logging import basicConfig, getLogger

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from icecream import ic

from abstract_solver import AbstractEquationSolver

class EquationSolverExp(AbstractEquationSolver):
    # @override
    def calc_fitness(self, w) -> np.ndarray:
        Y_sol = self.calcY(w)
        y_diff = np.gradient(Y_sol, self.x, axis=1)
        y_diff2 = np.gradient(y_diff, self.x, axis=1)
        fitEq= np.abs(
            y_diff2 + 25 * Y_sol
        )
        m = fitEq.shape[1]
        fitEq: np.ndarray = np.sqrt(np.power(fitEq, 2).sum(axis=1) / m)
        # fitEq = fitEq.sum(axis=1) / m
        fitEq = fitEq.reshape(-1) / fitEq.std()
        
        gamma = np.sin(self.current_generation / self.max_generation * 2 * np.pi)**2
        
        o = self.origin_points
        y_origin = (Y_sol[:, o[0]] + Y_sol[:, o[1]]) / 2
        # y_origin = w[:, 0]
        y_diff_origin = (y_diff[:, o[0]] + y_diff[:, o[1]]) / 2
        # y_diff_origin = (Y_sol[:, o[0]] + Y_sol[:, o[1]]) / 2 / self.h
        
        fitIni = gamma * np.abs(1 - y_origin) \
            + (1 - gamma) * np.abs(-y_diff_origin)
        fitIni = fitIni.reshape(-1)
        
        # ic(fitEq.mean())
        # ic(y_origin.mean())
        # ic(y_diff_origin.mean())
        return fitEq + fitIni

def main():
    basicConfig(
        format='\x1b[34m%(levelname)s\x1b[0m: [\x1b[32m%(funcName)s\x1b[0m] \x1b[36m%(message)s\x1b[0m',
        level=logging.INFO
    )
    
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    np.set_printoptions(formatter={'float': '{: 0.4f}'.format}) #桁を揃える
    logger = getLogger('Cos-solver')
    
    a = -1
    b = 1
    c = 9
    M = 258
    GENERATION = 2**14
    POPULATION = 258
    beta = 0.2
    mu = 0.05
    
    solver = EquationSolverExp(a, b, c, M, population=POPULATION, beta=beta, mu=mu, logger=logger)
    
    def analytical(x):
        return np.cos(5*x)
    
    st = time.time()
    
    # best_w = solver.solve(GENERATION)
    best_w = solver.solveWithWatching(GENERATION, analytical=analytical)
    
    et = time.time()
    
    logger.info(f"Finish solving. time: {et - st}")
    
    y_solve = solver.calcY(best_w)
    
    x = np.linspace(a, b, M)

    y_analytical = analytical(x)
    
    mse = solver.score(y_solve, y_analytical)
    logger.info(f"MSE = {mse}")
    
    axs: List[Axes]
    fig, axs = plt.subplots(1, 2)
    
    axs[0].plot(x, analytical(x), label='Analytical')
    axs[0].scatter(x[::6], y_solve[::6], s=12, marker='*', color='green', label='$y_{solve}$')
    axs[0].legend()
    history = solver.fitnessHistory
    axs[1].plot(range(len(history)), history)
    axs[1].set_title("Fitness")
    
    plt.show()

if __name__ == '__main__':
    main()
