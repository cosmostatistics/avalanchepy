from abc import ABC, abstractmethod
from typing import Callable, Tuple

import numpy as np


class MCMCUpdater(ABC):
    """Abstract class for updating the state of a sampler."""

    @abstractmethod
    def update(self, state: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        """Update the sampler state.

        Args:
            state (np.ndarray): Current state of the sampler.

        Returns:
            np.ndarray: New state of the sampler.
            str:        what action was taken (move, spawn, kill, etc)
            bool:       whether the state change was accepted
        """
        pass

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        return self.update(state)


class Select(MCMCUpdater):
    """MCMCUpdater for the standard case: mostly canonical moves of
    individual chains are performed, every few iterations, there is a
    spawn/kill step.

    Args:
        move (MCMCUpdater):         for the move steps
        spawn_kill (MCMCUpdater):   for the spawn/kill steps
        n_pos_updates_per_kill_spawn (int): number of move steps between any spawn/kill step
    """

    def __init__(self, move: MCMCUpdater, spawn_kill: MCMCUpdater, n_pos_updates_per_kill_spawn: int):
        self.move = move
        self.spawn_kill = spawn_kill
        self.counter = 0
        self.n_pos_updates_per_kill_spawn = n_pos_updates_per_kill_spawn

    def update(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        self.counter += 1
        if self.counter % self.n_pos_updates_per_kill_spawn == 0:  # kill or spawn
            return self.spawn_kill(pos)
        else:  # move
            return self.move(pos)


class CanonicalMove(MCMCUpdater):
    """performs a move of a random chain according to the Metropolis-Hastings
    algorithm for the canonical partition sum.

    Args:
        log_probability:       log_probability potential from the likelihood
        cov_move:   covariance matrix of the normal distribution used to propose
                    the new position of the thain
    """

    def __init__(self, log_probability: Callable, cov_move: np.ndarray):
        self.log_probability = log_probability
        self.cov_move = cov_move

    def update(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        index = int(np.random.randint(0, pos.shape[0]))
        theta_new = np.random.multivariate_normal(pos[index], self.cov_move)

        delta = -self.log_probability(theta_new) + self.log_probability(pos[index])
        if np.random.rand() < np.exp(-delta):
            pos[index] = theta_new
            return pos, index, "move", True
        else:
            return pos, index, "move", False


class StaticSpawnKill(MCMCUpdater):
    """spawn positions for new chains are sampled from a uniform distribution
    covering a given area, the chains to kill are selected at random.

    Args:
        log_probability:log_probability potential from the likelihood
        mu:             chemical potential
        spawn_range:    characterises area where chains may spawn (same
                        boundaries over all dimensions)
        n_dim:          dimension of the parameter space
    """

    def __init__(self, log_probability: Callable, mu: float, spawn_range: Tuple[float, float], n_dim: int):
        self.log_probability = log_probability
        self.mu = mu
        self.spawn_range = spawn_range
        self.n_dim = n_dim

    def update(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        if np.random.rand() < 0.5:
            return self.spawn(pos)
        else:
            return self.kill(pos)

    def spawn(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        theta_new = np.random.uniform(self.spawn_range[0], self.spawn_range[1], self.n_dim)
        delta = self.mu + self.log_probability(theta_new)
        acceptance_probability = (
            np.exp(delta) / (pos.shape[0] + 1) * (self.spawn_range[1] - self.spawn_range[0]) ** self.n_dim
        )

        if np.random.rand() < acceptance_probability:
            pos = np.vstack([pos, theta_new])
            return pos, -1, "spawn", True
        else:
            return pos, -1, "spawn", False

    def kill(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        kill_idx = int(np.random.randint(0, pos.shape[0]))
        delta = -self.mu - self.log_probability(pos[kill_idx])
        acceptance_probability = (
            np.exp(delta)
            * pos.shape[0]
            / (self.spawn_range[1] - self.spawn_range[0]) ** self.n_dim
            / np.float64(np.all((pos[kill_idx] > self.spawn_range[0]) & (pos[kill_idx] < self.spawn_range[1])))
        )

        if np.random.rand() < acceptance_probability:
            pos = np.delete(pos, kill_idx, axis=0)
            return pos, kill_idx, "kill", True
        else:
            return pos, kill_idx, "kill", False


class GaussianStaticSpawnKill(MCMCUpdater):
    """spawn positions for new chains are dsamplern from a gaussian distribution
    with given covariance around the origin, the chains to kill are selected at random.

    Args:
        log_probability: log_probability potential from the likelihood
        mu:              chemical potential
        spawn_cov:       characterises gaussians where chains may spawn
        n_dim:           dimension of the parameter space
    """

    def __init__(self, log_probability: Callable, mu: float, cov_spawn: np.ndarray, mean: np.ndarray, n_dim: int):
        self.log_probability = log_probability
        self.mu = mu
        self.cov_spawn = cov_spawn
        self.n_dim = n_dim
        self.mean = mean

        self.cov_spawn_inv = np.linalg.inv(cov_spawn)
        self.norm_factor = 1 / np.sqrt(np.linalg.det(2 * np.pi * self.cov_spawn))
        if cov_spawn.shape != (n_dim, n_dim):
            raise ValueError("spawn_cov must be a square matrix of dimension n_dim")
        if mean.shape != (n_dim,):
            raise ValueError("mean must be a vector of dimension n_dim")

    def update(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        if np.random.rand() < 0.5:
            return self.spawn(pos)
        else:
            return self.kill(pos)

    def spawn(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        theta_new = np.random.multivariate_normal(self.mean, self.cov_spawn)

        delta = self.mu + self.log_probability(theta_new)
        acceptance_probability = (
            np.exp(delta)
            / (pos.shape[0] + 1)
            / self.norm_factor
            / np.exp(-0.5 * np.dot(theta_new - self.mean, np.dot(self.cov_spawn_inv, theta_new - self.mean)))
        )

        if np.random.rand() < acceptance_probability:
            pos = np.vstack([pos, theta_new])
            return pos, -1, "spawn", True
        else:
            return pos, -1, "spawn", False

    def kill(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        kill_idx = int(np.random.randint(0, pos.shape[0]))
        delta = -self.mu - self.log_probability(pos[kill_idx])
        acceptance_probability = (
            np.exp(delta)
            * pos.shape[0]
            * self.norm_factor
            * np.exp(-0.5 * np.dot(pos[kill_idx] - self.mean, np.dot(self.cov_spawn_inv, pos[kill_idx] - self.mean)))
        )

        if np.random.rand() < acceptance_probability:
            pos = np.delete(pos, kill_idx, axis=0)
            return pos, kill_idx, "kill", True
        else:
            return pos, kill_idx, "kill", False


class ProximitySpawnKill(MCMCUpdater):
    """spawn positions for new chains are sampled from a gaussian distribution
    with given covariance around the origin, the chains to kill are selected
    based on the proximity of the chains.

    Args:
        log_probability: log_probability potential from the likelihood
        mu:              chemical potential
        cov_proximity:   characterises gaussians where chains may spawn
    """

    def __init__(self, log_probability: Callable, mu: float, cov_proximity: np.ndarray):
        self.log_probability = log_probability
        self.mu = mu
        self.cov_proximity = cov_proximity
        self.cov_proximity_inv = np.linalg.inv(cov_proximity)
        self.det_cov_proximity = np.linalg.det(cov_proximity)
        self.n_dim = cov_proximity.shape[0]

    def update(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        if np.random.rand() < 0.5:
            return self.spawn(pos)
        else:
            return self.kill(pos)

    def calc_P_C(self, pos: np.ndarray) -> np.ndarray:
        distances = pos[None, :, :] - pos[:, None, :]
        P_C = (
            1
            / np.sqrt((2 * np.pi) ** self.n_dim * self.det_cov_proximity)
            * np.exp(-1 / 2 * np.einsum("ij,kli,klj->kl", self.cov_proximity_inv, distances, distances, optimize=True))
        )
        return np.array(P_C)

    def kill(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        P_C = self.calc_P_C(pos)
        p = np.sum(P_C, axis=1) - np.diagonal(P_C)
        p /= np.sum(p)
        kill_idx = int(np.random.choice(range(pos.shape[0]), p=p))
        if np.random.rand() < np.exp(-self.mu - self.log_probability(pos[kill_idx])) * (np.sum(P_C) - np.trace(P_C)) / (
            pos.shape[0] - 1
        ):
            return np.delete(pos, kill_idx, axis=0), kill_idx, "kill", True
        else:
            return pos, kill_idx, "kill", False

    def spawn(self, pos: np.ndarray) -> Tuple[np.ndarray, int, str, bool]:
        theta_new = np.random.multivariate_normal(pos[np.random.randint(pos.shape[0])], self.cov_proximity)
        P_C = self.calc_P_C(np.vstack([pos, theta_new]))
        if np.random.rand() < np.exp(self.mu + self.log_probability(theta_new)) * pos.shape[0] / (
            np.sum(P_C) - np.trace(P_C)
        ):
            return np.vstack([pos, theta_new]), -1, "spawn", True
        else:
            return pos, -1, "spawn", False
