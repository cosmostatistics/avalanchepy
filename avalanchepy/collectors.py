from abc import ABC, abstractmethod
from typing import Callable

import numpy as np


class Collector(ABC):
    """Abstract class for collecting data from the MCMC simulation.

    Args:
        name (str): name of the collector/observable
        expected_length (int): expected length of the measured data
        burn_in (int): number of steps to be discarded at the beginning of run
        thinning (int): number of steps to skip between each measurement
    """

    def __init__(self, name: str, expected_length: int, burn_in: int = 0, thinning: int = 1):
        self.name = name
        self.burn_in = burn_in
        self.thinning = thinning

        self.values = np.zeros(expected_length)
        self.abbreviation = name

    @abstractmethod
    def collect(self, state: np.ndarray, i: int, touched_chain: int, action: str, success: bool) -> None:
        return None

    def get(self, i: int) -> np.ndarray:
        return self.values[:i]

    def get_clean(self, i: int) -> np.ndarray:
        return self.values[self.burn_in :: self.thinning]

    def get_recent_average(self, i: int, window_size: int = 100) -> np.float64:
        return np.mean(self.values[i - window_size : i])


class Samples(Collector):
    """collects the samples every few steps after burn-in, not ordered
    by chains"""

    def __init__(self, expected_length: int, burn_in: int, thinning: int, expected_n_chains: int, n_dims: int):
        super().__init__("samples", expected_length, burn_in, thinning)
        # self.values = np.zeros((expected_n_chains * (expected_length-burn_in) // thinning, n_dims))
        # self.values = np.zeros((expected_length*expected_n_chains, n_dims))
        self.values = np.zeros((expected_length, n_dims))
        self.last_idx = 0

    def collect(self, state: np.ndarray, i: int, touched_chain: int, action: str, success: bool) -> None:
        # print(self.values.shape, state.shape, i)
        if action == "kill" and success:
            self.values[self.last_idx] = state[np.random.randint(state.shape[0])]
        else:
            self.values[self.last_idx] = state[touched_chain]
        self.last_idx += 1

    def get(self, i: int) -> np.ndarray:
        return self.values[: self.last_idx]

    def get_clean(self, i: int) -> np.ndarray:
        return self.get(i)[self.burn_in :: self.thinning]


class Hamiltonian(Collector):
    """measures the energy of the system, i.e. the Hamiltonian"""

    def __init__(
        self,
        expected_length: int,
        burn_in: int,
        thinning: int,
        log_probability: Callable[[np.ndarray], np.ndarray[float]],
    ):
        super().__init__("hamiltonian", expected_length, burn_in, thinning)
        self.hamiltonian = lambda state: np.sum(log_probability(state))
        self.abbreviation = "H"

    def collect(self, state: np.ndarray, i: int, touched_chain: int, action: str, success: bool) -> None:
        self.values[i] = self.hamiltonian(state)
        return None


class ChainSize(Collector):
    """measures the number of chains"""

    def __init__(self, expected_length: int, burn_in: int, thinning: int):
        super().__init__("n_chains", expected_length, burn_in, thinning)

        self.values = np.zeros(expected_length, dtype=int)
        self.abbreviation = "N"

    def collect(self, state: np.ndarray, i: int, touched_chain: int, action: str, success: bool) -> None:
        self.values[i] = state.shape[0]
        return None


class AvgSecondMoment(Collector):
    """measures the second moment averaged over all chains"""

    def __init__(self, expected_length: int, burn_in: int, thinning: int, n_dims: int) -> None:
        super().__init__("avg_second_moment", expected_length, burn_in, thinning)

        self.n_dims = n_dims
        self.values = np.zeros((expected_length, n_dims, n_dims), dtype=np.float64)

    def collect(self, pos: np.ndarray, i: int, touched_chain: int, action: str, success: bool) -> None:
        self.values[i] = 1 / pos.shape[0] * np.einsum("li,lj->ij", pos, pos)
