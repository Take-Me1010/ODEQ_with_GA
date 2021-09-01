
from typing import Any, Tuple
import numpy as np

class GA:
    def __init__(self, n_population: int, mu: float = 0.01, seed: Any = 64) -> None:
        self.rng = np.random.default_rng(seed)
        # self.population = self.rng.integers(2, size=(n_population, 5))
        self.population = self.rng.random(size=(n_population, 5)) <= 0.5
        self.mu = mu
    
    def pop2int(self, pop: np.ndarray) -> np.ndarray:
        ''' popを整数値の配列に変換して返す '''
        s = pop.shape
        b2i = 2**np.arange(s[1]-1, -1, -1).reshape((1, -1)).repeat(s[0], axis=0)
        return (pop*b2i).sum(axis=1)

    def fitness(self, pop: np.ndarray) -> np.ndarray:
        #ref: https://living-sun.com/ja/python/718584-binary-numpy-array-to-list-of-integers-python-numpy-binary.html
        int_values = self.pop2int(pop)
        return np.abs(int_values - 5)

    def bestIndividuals(self, pop: np.ndarray, fits: np.ndarray, k: int = 1,) -> Tuple[np.ndarray, np.ndarray]:
        ''' 集団popから上位k個体を抽出して返す '''
        ranks = np.argsort(fits)
        pop_sorted = pop[ranks]
        fits_sorted = fits[ranks]
        return pop_sorted[0:k, :], fits_sorted[0:k]

    def selection(self, pop: np.ndarray, fits: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_population = pop.shape[0]
        next_pop, next_fits = self.bestIndividuals(pop, fits, k=n_population//2)
        return next_pop, next_fits
    
    def crossover(self, pop: np.ndarray) -> np.ndarray:
        n_population, n_gene = pop.shape
        id_all = self.rng.choice(n_population, n_population, replace=False)
        
        n_couple = n_population//2
        id_dad = id_all[:n_couple]
        id_mom = id_all[n_couple:]
        
        offsprings = []
        
        for dad_index, mom_index in zip(id_dad, id_mom):
            dad = pop[dad_index]
            mom = pop[mom_index]
            
            doSwap = self.rng.integers(2, size=n_gene) == 1
            
            boy, girl = dad.copy(), mom.copy()
            boy[doSwap], girl[doSwap] = girl[doSwap], boy[doSwap]
        
            offsprings.append(boy)
            offsprings.append(girl)
        
        return np.vstack((pop, np.array(offsprings)))
    
    def mutation(self, pop: np.ndarray) -> np.ndarray:
        doMutation = self.rng.uniform(0, 1, size=pop.shape) < self.mu
        return np.where(doMutation, 1 - pop, pop)
    
    def solve(self, generation: int) -> int:
        for g in range(generation):
            pop = self.population
            fits = self.fitness(pop)
            
            mean = fits.mean()
            min_ = fits.min()
            print(f"--- Generation: {g+1} ---")
            print(f"\tmean: {mean}")
            print(f"\tmin: {min_}")
            
            pop, fits = self.selection(pop, fits)
            
            pop = self.crossover(pop)
            
            pop = self.mutation(pop)
            
            self.population = pop
        
        best, _ = self.bestIndividuals(pop, fits=self.fitness(pop), k=1)
        return self.pop2int(best)[0]

def main():
    solver = GA(10, 0.01)
    
    ans = solver.solve(100)
    
    print(f"x = {ans}")

if __name__ == '__main__':
    main()
