
import math
from typing import Any, Callable, Generator, List, Tuple, Optional, Union
from logging import INFO, Logger, basicConfig, getLogger

import numpy as np
from numpy import random

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

try:
    from icecream import ic
except ImportError:
    def ic(*args):
        print("ic: " + str(args))
        return args

class AbstractEquationSolver:
    fitnessHistory: List[float]
    w: np.ndarray
    beta: float
    mu: float
    x: np.ndarray
    vander: np.ndarray
    origin_points: Tuple[int, int]              # self.xにおいて原点に近い2点のインデックス
    current_generation: Union[int, None]         # 現在の世代を保持
    max_generation: Union[int, None]            # solveを呼び出した際に最大世代数を保持
    logger: Logger
    rng: random.Generator
    def __init__(self, a: float, b: float, c: int, m: int = 258, population: int = 258, beta: float = 0.6, mu: float = 0.1, logger: Optional[Logger] = None, seed: Any = 64) -> None:
        self.rng = random.default_rng(seed)
        
        self.a = a
        self.b = b
        self.c = c
        self.m = m
        self.beta = beta
        self.mu = mu
        
        if a > b or (a > 0 and b > 0):
            raise ValueError("Set args [a, b] as a < 0 < b")
        
        self.x = np.linspace(a, b, m)
        self.h = (b - a) / (m - 1)
        i = - a / self.h
        self.origin_points = (math.floor(i), math.ceil(i))
        self.vander = self._getVander(c)
        
        self.w = self._init_w(population, c)
        
        self.logger = logger if logger else getLogger('Solver')
        
        self.fitnessHistory = []
        
        
        
    def _init_w(self, n_population: int, n_coef: int) -> np.ndarray:
        return self.rng.uniform(0, 1, size=(n_population, n_coef))
        
    def _getVander(self, c: int) -> np.ndarray:
        ''' Taylor Matrixの転置を取得する。cは係数の数(c-1が最大の次数になる) '''
        return np.vander(self.x.reshape(-1), increasing=True, N=c).T
    
    
    def calcY(self, w: np.ndarray) -> np.ndarray:
        ''' 集団を表すwからY_solveを計算する '''
        return w.dot(self.vander)
        
        
    def _iterator(self, max_generation: int):
        ''' 世代番号を出力するイテレーター。 '''
        self.max_generation = max_generation
        for g in range(max_generation):
            self.current_generation = g
            yield g
    
    def calc_fitness(self, w: np.ndarray) -> np.ndarray:
        ''' 集団wから、適合度を表すベクトル(shape=(-1, 1))を計算して返す。この値が小さいほど優秀な個体である。 '''
        raise NotImplementedError()
    
    def bestIndividuals(self, w: np.ndarray, fits: np.ndarray, k: int = 1,) -> Tuple[np.ndarray, np.ndarray]:
        ''' 集団wから上位k個体を抽出して返す '''
        ranks = np.argsort(fits)
        w_sorted = w[ranks]
        fits_sorted = fits[ranks]
        return w_sorted[0:k, :], fits_sorted[0:k]
    
    def _selection(self, w: np.ndarray, fits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        ''' 集団及びその評価値の集合から、次世代生存集団及びその評価値を返す '''
        pop = w.shape[0]
        next_w, next_fits = self.bestIndividuals(w, fits, k=pop//2)
        return next_w, next_fits
    
    def _crossover(self, w: np.ndarray) -> np.ndarray:
        ''' 次世代集団を生成する '''
        now_pop = w.shape[0]
        
        id_all = self.rng.choice(now_pop, now_pop, replace=False)
        
        n_couple = now_pop//2
        id_dad = id_all[:n_couple]
        id_mom = id_all[n_couple:]
        
        candidates = []
        beta = self.beta
        for dad_index, mom_index in zip(id_dad, id_mom):
            dad = w[dad_index]
            mom = w[mom_index]
            
            candidates.append(
                beta * (dad - mom) + dad
            )
            candidates.append(
                beta * (mom - dad) + mom
            )
            
        candidates = np.array(candidates)
        
        return np.vstack((w, candidates))
    
    def _mutation(self, w: np.ndarray) -> np.ndarray:
        ''' 突然変異を起こした集団を返す '''
        size = w.shape
        IsMutated = self.rng.uniform(0, 1, size) < self.mu
        mutation = w + self.rng.normal(0, 1, size)
        next_w = np.where(IsMutated, mutation, w)
        return next_w
    
    def _iteration(self):
        ''' 各世代での処理。 '''
        
        g = self.current_generation
        if not g % 100:
            self.logger.info(f"Generation: {g} (Done {g / self.max_generation * 100: .3f}%)")
        
        w = self.w.copy()
        
        fits = self.calc_fitness(w)
        
        w, fits = self._selection(w, fits)
        
        # best_w = w[0]
        best_fits = fits[0]
        
        self.fitnessHistory.append(best_fits)
        
        w = self._crossover(w)
        
        w = self._mutation(w)
        
        self.w = w.copy()
        
        return w
    
    def solve(self, genetration: int) -> np.ndarray:
        ''' ベストな個体(係数ベクトル)を返す '''
        w = self.w
        for g in self._iterator(genetration):
            w = self._iteration()
    
        w, fits = self.bestIndividuals(w, fits=self.calc_fitness(w), k=1)
        return w[0]
    
    def solveWithWatching(self, generation: int, analytical: Optional[Callable[[np.ndarray], np.ndarray]] = None, lim_fitness: Optional[float] = None) -> np.ndarray:
        w = self.w
        self.current_generation = 0
        self.max_generation = generation
        w_best, fit_best = self.bestIndividuals(w, fits=self.calc_fitness(w), k=1)
        x = self.x
        
        axSolved: Axes
        fig, axSolved = plt.subplots(1, 1)
        
        # axSolved = axs[0]
        axSolved.set_title('Analytical answer and best $y_{solve}$')
        axSolved.set_xlim((self.a, self.b))
        axSolved.set_xlabel('x')
        axSolved.set_ylabel('y')
        # 解析解が与えられていたらプロットして最大最小を設定する
        if analytical:
            y_analytical = analytical(x)
            axSolved.plot(x, y_analytical)
            axSolved.set_ylim((y_analytical.min()-0.5, y_analytical.max()+0.5))
        # 初期集団における最良個体を表示
        y_best = self.calcY(w_best).reshape(-1)
        line_solved, = axSolved.plot(x, y_best)
        
        for g in self._iterator(generation):
            w = self._iteration()
            
            if not g%10:
                w_best, _ = self.bestIndividuals(w, fits=self.calc_fitness(w), k=1)
                y_best = self.calcY(w_best)
                line_solved.set_data(x, y_best)
                plt.pause(0.01)
        
        w, _ = self.bestIndividuals(w, fits=self.calc_fitness(w), k=1)
        return w[0]

    def score(self, y_solve: Optional[np.ndarray], y_analytical: np.ndarray) -> float:
        if y_solve is None:
            w_best = self.bestIndividuals(self.w, self.calc_fitness(self.w), k=1)
            y_solve = self.calcY(w_best)

        mse = np.sqrt(np.power(y_solve - y_analytical, 2).sum() / self.m)
        return mse
