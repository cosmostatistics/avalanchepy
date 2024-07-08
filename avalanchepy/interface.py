from typing import Callable

import numpy as np

from .collectors import AvgSecondMoment, ChainSize, Hamiltonian, Samples
from .sampler import macro_sampler
from .updaters import CanonicalMove, GaussianStaticSpawnKill, ProximitySpawnKill, Select, StaticSpawnKill


class Result(dict):
    """Object containing the results of a run of the sampler.
    It's a dictionary with some additional methods for printing summaries and accessing the results more conveniently.
    """

    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, values) -> None:
        super().__init__()
        self.__dict__ = self

        for key, value in values.items():
            self[key] = value

        self.n_chains_avg = self.n_chains_clean.mean()
        self.log_evidence = np.log(self.n_chains_avg) - self.mu
        self.log_evidence_err = self.n_chains_clean.std() / np.sqrt(len(self.n_chains_clean))

    def print_summary(self) -> None:
        """Prints a summary of the runs results."""
        print(self.summary())

    def summary(self) -> str:
        """Returns a summary of the runs results.

        Returns:
            str: Summary of results
        """
        summary_str = ""
        summary_str += "mode: " + str(self.mode) + "\n"
        summary_str += "n_dim: " + str(self.n_dim) + "\n"
        summary_str += "n_samples: " + str(len(self.samples_clean)) + "\n"
        summary_str += "<N>: " + str(self.n_chains_avg) + "\n"
        summary_str += "Var(N): " + str(self.n_chains.std()) + "\n"
        summary_str += "log_evidence: " + str(self.log_evidence) + "\n"
        summary_str += "log_evidence_err: " + str(self.log_evidence_err) + "\n"
        summary_str += "evidence: " + str(np.exp(self.log_evidence)) + "\n"
        summary_str += "evidence_err: " + str(np.exp(self.log_evidence) * self.log_evidence_err) + "\n"

        return summary_str


def run(log_probability: Callable, **kwargs) -> Result:
    """
    Performs a run of Avalanche Sampling. The idea is that only the log_probability potential needs to be passed
    as well as a 'mode' in which the sampler operates.

    Modes:
        - `"static_spawn"` : Positions for new chains are sampled from a static uniform distribution.
            Keyword arguments:

            | Name         | Type         | Description                                     | Default     |
            |--------------|--------------|-------------------------------------------------|-------------|
            | `spawn_range`| `tuple`      | Range from which to sample new chains.          | `(-3, 3)`   |

        - `"gaussian_static_spawn"` : Positions for new chains are sampled from a static Gaussian distribution.
            Keyword arguments:

            | Name      | Type         | Description                                                | Default            |
            |-----------|--------------|------------------------------------------------------------|--------------------|
            | `cov_spawn` | `np.ndarray` | Covariance matrix for the proposal distribution when spawning new chains. | `np.eye(n_dim)`    |
            | `mean`    | `np.ndarray` | Mean of the proposal distribution when spawning new chains.| `np.zeros(n_dim)`  |

        - `"proximity_spawn"` : Positions for new chains are close to existing chains, chains are more likely to be killed when close to others.
            Keyword arguments:

            | Name          | Type         | Description                                                | Default         |
            |---------------|--------------|------------------------------------------------------------|-----------------|
            | `cov_proximity`| `np.ndarray` | Covariance matrix for the proposal distribution when spawning new chains. | `np.eye(n_dim)`  |


    Alternatively, keyword arguments allow to pass other arguments to the function, possibly overwriting those set by the mode.

    Args:
        log_probability (function): The logarithm of the probability function to be used. Also known as the chi^2 potential.
        **kwargs: Additional keyword arguments that can be passed to the function.

    optional keyword arguments:

    | Name                          | Type         | Description                                                                  | Default                  |
    |-------------------------------|--------------|------------------------------------------------------------------------------|--------------------------|
    | `mode`                        | `str`        | See above.                                                                   | `"proximity_spawn"`      |
    | `mu`                          | `float`      | Chemical potential.                                                          | `3`                      |
    | `n_dim`                       | `int`        | Number of dimensions.                                                        | `2`                      |
    | `n_initial_particles`         | `int`        | Number of chains to start with.                                              | `100`                    |
    | `n_total_steps`               | `int`        | Number of total steps to take.                                               | `10000`                  |
    | `n_thinning`                  | `int`        | Number of steps to skip before storing a sample.                             | `10`                     |
    | `n_burn_in_steps`             | `int`        | Number of steps to skip before starting to store samples.                    | `n_total_steps/3`        |
    | `n_pos_updates_per_kill_spawn`| `int`        | Number of position updates to perform before trying to spawn or kill a chain.| `5`                      |
    | `cov_move`                    | `np.ndarray` | Covariance matrix for the proposal distribution when moving particles.       | `np.eye(n_dim)`          |


    Returns:
        result (Result): The result of the Avalanche Sampling run.
    """
    # ###
    # DEFAULTS
    # ###

    mode = kwargs.pop("mode", "proximity_spawn")

    n_dim = kwargs.pop("n_dim", 2)
    n_initial_particles = kwargs.pop("n_initial_particles", 30)
    n_total_steps = kwargs.pop("n_total_steps", 100000)
    n_skip_steps = kwargs.pop("n_thinning", 10)
    n_burn_in_steps = kwargs.pop("n_burn_in_steps", int(n_total_steps / 10))
    n_pos_updates_per_kill_spawn = kwargs.pop("n_pos_updates_per_kill_spawn", 3)
    cov_move = kwargs.pop("cov_move", np.eye(n_dim) / n_dim)
    progress_bar = kwargs.pop("progress_bar", True)

    log_evidence = kwargs.pop("log_evidence", None)
    n_particles_wanted = kwargs.pop("n_particles", None)

    if log_evidence:
        if not n_particles_wanted:
            mu = kwargs.pop("mu", np.log(n_initial_particles) - log_evidence)
        else:
            mu = kwargs.pop("mu", np.log(n_particles_wanted) - log_evidence)
    else:
        mu = kwargs.pop("mu", 3)
        if n_particles_wanted:
            raise ValueError(
                "Error: no evidence provided, cannot compute chemical potential to reach desired number of particles"
            )
    standard_move = CanonicalMove(log_probability, cov_move)
    # mode specific arguments
    spawn_range = kwargs.pop("spawn_range", (-1, 1))
    initial_positions = kwargs.pop(
        "initial_positions", np.random.uniform(spawn_range[0], spawn_range[1], (n_initial_particles, n_dim))
    )

    macro_sampler_hyperparameters = {
        "n_total_steps": n_total_steps,
        "progress_bar": progress_bar,
        "initial_positions": initial_positions,
    }

    # ###
    # MODES
    # ###

    static_spawn_kill = StaticSpawnKill(log_probability, mu, spawn_range, n_dim)
    mode_updater = {"static_spawn": Select(standard_move, static_spawn_kill, n_pos_updates_per_kill_spawn)}

    mean = kwargs.pop("mean", np.zeros(n_dim))
    cov_spawn = kwargs.pop("cov_spawn", np.eye(n_dim) / n_dim**2)
    gaussian_static_spawn_kill = GaussianStaticSpawnKill(log_probability, mu, cov_spawn, mean, n_dim)
    mode_updater["gaussian_static_spawn"] = Select(
        standard_move, gaussian_static_spawn_kill, n_pos_updates_per_kill_spawn
    )

    cov_proximity = kwargs.pop("cov_proximity", np.eye(n_dim))
    proximity_spawn_kill = ProximitySpawnKill(log_probability, mu, cov_proximity)
    mode_updater["proximity_spawn"] = Select(standard_move, proximity_spawn_kill, n_pos_updates_per_kill_spawn)

    # ###
    # COLLECTORS
    # ###

    samples_collector = Samples(n_total_steps, n_burn_in_steps, n_skip_steps, n_initial_particles, n_dim)
    Hamiltonian(n_total_steps, n_burn_in_steps, n_skip_steps, log_probability)
    chainsize_collector = ChainSize(n_total_steps, n_burn_in_steps, n_skip_steps)
    AvgSecondMoment(n_total_steps, n_burn_in_steps, n_skip_steps, n_dim)
    collectors = kwargs.pop("collectors", [chainsize_collector, samples_collector])

    # ###
    # RUN
    # ###

    # check if keyword arguments are valid
    if kwargs:
        raise TypeError("The following keyword arguments are unknown: ", kwargs.keys())

    if mode not in mode_updater:
        raise ValueError("Mode '" + str(mode) + "' not recognised")

    result = macro_sampler(mode_updater[mode], collectors=collectors, **macro_sampler_hyperparameters)

    return Result({**result, **macro_sampler_hyperparameters, **kwargs, "n_dim": n_dim, "mu": mu, "mode": mode})
