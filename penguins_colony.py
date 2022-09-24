from src.optimization_algorithm.meta import OptimizationAlgorithmMeta
import numpy as np
from numpy import exp
from numpy.random import uniform
from .utils import (merge, compute_binary_position, compute_binary_position_tanh,
                    compute_binary_position_v)
import warnings


class PenguinsColonyMerge(OptimizationAlgorithmMeta):
    def __init__(
            self,
            population,
            function,
            max_iter,
            lb=None,
            ub=None,
            no_change_iteration=None,
    ):
        super().__init__(
            population, function, lb, ub, max_iter, no_change_iteration, feature_selection=True)

        self.current_it = 0

    warnings.filterwarnings('ignore')

    def sort_agents(self):
        order = self._sorted_fitness_args
        self._agents = self._agents[order]
        self._fitness = np.array(self._fitness)[order.tolist()]

    def search(self):
        best_penguin = self._agents[0]
        polygon_grid = np.absolute(best_penguin - self._agents)
        movement_parameter = 1
        r = uniform()
        temperature = 0 if r > 1 else 1
        temperature_profile = temperature - self.max_iter / (self.current_it - self.max_iter)

        for i in range(1, self.n):
            rand = np.random.random(size=self.dimension)
            a = movement_parameter * (temperature_profile + polygon_grid[i]) * rand - temperature_profile

            f = np.random.uniform(2, 3, self.dimension)
            l = np.random.uniform(1.5, 2, self.dimension)
            social_forces = np.absolute(f * exp(-self.current_it / l) - exp(-self.current_it))

            c = np.random.random(size=self.dimension)
            distance = np.absolute(social_forces * best_penguin - c * self._agents[i])

            self._agents[i] = merge(self._agents[i], best_penguin)
            self._agents[i] = self.force_bounds(self._agents[i])
            self._fitness[i] = self.ff(self._agents[i])

    def _one_iter(self):
        self.sort_agents()
        self.search()
        self.get_best()
        # print(self.current_it)
        # print(self._fitness[0])


class PenguinsColonySigma(OptimizationAlgorithmMeta):
    def __init__(
            self,
            population,
            function,
            max_iter,
            lb=None,
            ub=None,
            no_change_iteration=None,
    ):
        super().__init__(
            population, function, lb, ub, max_iter, no_change_iteration, feature_selection=True)

        self.current_it = 0

    warnings.filterwarnings('ignore')

    def sort_agents(self):
        order = self._sorted_fitness_args
        self._agents = self._agents[order]
        self._fitness = np.array(self._fitness)[order.tolist()]

    def calculate_vll(self, worst_penguin, local_leader, best_penguin):
        r1 = np.random.uniform(0, 1, self.dimension)
        r2 = np.random.uniform(0, 1, self.dimension)

        return (
                (best_penguin - worst_penguin) * r1 +
                (local_leader - worst_penguin) * r2
        )

    def search(self):
        best_penguin = self._agents[0]
        polygon_grid = np.absolute(best_penguin - self._agents)
        movement_parameter = 1
        r = uniform()
        temperature = 0 if r > 1 else 1
        temperature_profile = temperature - self.max_iter / (self.current_it - self.max_iter)

        for i in range(1, self.n):
            rand = np.random.random(size=self.dimension)
            a = movement_parameter * (temperature_profile + polygon_grid[i]) * rand - temperature_profile

            f = np.random.uniform(2, 3, self.dimension)
            l = np.random.uniform(1.5, 2, self.dimension)
            social_forces = np.absolute(f * exp(-self.current_it / l) - exp(-self.current_it))

            c = np.random.random(size=self.dimension)
            distance = np.absolute(social_forces * best_penguin - c * self._agents[i])

            v = self.calculate_vll(self._agents[self.n - 1], self._agents[i], self._agents[0])
            self._agents[i] = compute_binary_position(self.dimension, v)
            self._agents[i] = self.force_bounds(self._agents[i])
            self._fitness[i] = self.ff(self._agents[i])

    def _one_iter(self):
        self.sort_agents()
        self.search()
        self.get_best()
        # print(self.current_it)
        # print(self._fitness[0])


class PenguinsColonyV(OptimizationAlgorithmMeta):
    def __init__(
            self,
            population,
            function,
            max_iter,
            lb=None,
            ub=None,
            no_change_iteration=None,
    ):
        super().__init__(
            population, function, lb, ub, max_iter, no_change_iteration, feature_selection=True)

        self.current_it = 0

    warnings.filterwarnings('ignore')

    def sort_agents(self):
        order = self._sorted_fitness_args
        self._agents = self._agents[order]
        self._fitness = np.array(self._fitness)[order.tolist()]

    def calculate_vll(self, worst_penguin, local_leader, best_penguin):
        r1 = np.random.uniform(0, 1, self.dimension)
        r2 = np.random.uniform(0, 1, self.dimension)

        return (
                (best_penguin - worst_penguin) * r1 +
                (local_leader - worst_penguin) * r2
        )

    def search(self):
        best_penguin = self._agents[0]
        polygon_grid = np.absolute(best_penguin - self._agents)
        movement_parameter = 1
        r = uniform()
        temperature = 0 if r > 1 else 1
        temperature_profile = temperature - self.max_iter / (self.current_it - self.max_iter)

        for i in range(1, self.n):
            rand = np.random.random(size=self.dimension)
            a = movement_parameter * (temperature_profile + polygon_grid[i]) * rand - temperature_profile

            f = np.random.uniform(2, 3, self.dimension)
            l = np.random.uniform(1.5, 2, self.dimension)
            social_forces = np.absolute(f * exp(-self.current_it / l) - exp(-self.current_it))

            c = np.random.random(size=self.dimension)
            distance = np.absolute(social_forces * best_penguin - c * self._agents[i])

            v = self.calculate_vll(self._agents[self.n - 1], self._agents[i], self._agents[0])
            self._agents[i] = compute_binary_position_v(self.dimension, v)
            self._agents[i] = self.force_bounds(self._agents[i])
            self._fitness[i] = self.ff(self._agents[i])

    def _one_iter(self):
        self.sort_agents()
        self.search()
        self.get_best()
        # print(self.current_it)
        # print(self._fitness[0])


class PenguinsColonyTanh(OptimizationAlgorithmMeta):
    def __init__(
            self,
            population,
            function,
            max_iter,
            lb=None,
            ub=None,
            no_change_iteration=None,
    ):
        super().__init__(
            population, function, lb, ub, max_iter, no_change_iteration, feature_selection=True)

        self.current_it = 0

    warnings.filterwarnings('ignore')

    def sort_agents(self):
        order = self._sorted_fitness_args
        self._agents = self._agents[order]
        self._fitness = np.array(self._fitness)[order.tolist()]

    def calculate_vll(self, worst_penguin, local_leader, best_penguin):
        r1 = np.random.uniform(0, 1, self.dimension)
        r2 = np.random.uniform(0, 1, self.dimension)

        return (
                (best_penguin - worst_penguin) * r1 +
                (local_leader - worst_penguin) * r2
        )

    def search(self):
        best_penguin = self._agents[0]
        polygon_grid = np.absolute(best_penguin - self._agents)
        movement_parameter = 1
        r = uniform()
        temperature = 0 if r > 1 else 1
        temperature_profile = temperature - self.max_iter / (self.current_it - self.max_iter)

        for i in range(1, self.n):
            rand = np.random.random(size=self.dimension)
            a = movement_parameter * (temperature_profile + polygon_grid[i]) * rand - temperature_profile

            f = np.random.uniform(2, 3, self.dimension)
            l = np.random.uniform(1.5, 2, self.dimension)
            social_forces = np.absolute(f * exp(-self.current_it / l) - exp(-self.current_it))

            c = np.random.random(size=self.dimension)
            distance = np.absolute(social_forces * best_penguin - c * self._agents[i])

            v = self.calculate_vll(self._agents[self.n - 1], self._agents[i], self._agents[0])
            self._agents[i] = compute_binary_position_tanh(self.dimension, v)
            self._agents[i] = self.force_bounds(self._agents[i])
            self._fitness[i] = self.ff(self._agents[i])

    def _one_iter(self):
        self.sort_agents()
        self.search()
        self.get_best()
        # print(self.current_it)
        # print(self._fitness[0])
