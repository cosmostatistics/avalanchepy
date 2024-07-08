import contextlib

import numpy as np
from tqdm.auto import trange

from .collectors import Collector
from .updaters import MCMCUpdater


def macro_sampler(
    updater: MCMCUpdater,
    n_total_steps: int,
    initial_positions: np.ndarray,  # of the chains
    collectors: list[Collector],  # list of collectors
    progress_bar: bool = True,  # whether to show a progress bar
    info_window_size: int = 500,  # how many steps should be considered for the progress bar
) -> dict:
    """Runs the sampling at the lowest level of abstraction.

    Args:
        updater (MCMCUpdater): updates the state of the sampler in each step.
        n_total_steps (int): includes move and kill/spawn steps.
        initial_positions (np.ndarray): of the different chains.
        collectors (list[Collector]): used to measure observables between steps.
        progress_bar (bool): (using tqdm)
        info_window_size (int): for the progress bar.

    Returns:
        results (dict): contains the collected values for the observables and acceptance rates.
    """

    pos = initial_positions  # shape: (#chains, #dimensions)

    rates = {"move": [0, 0], "kill": [0, 0], "spawn": [0, 0]}

    t = trange(n_total_steps, desc="init", leave=True) if progress_bar else range(n_total_steps)

    for i in t:
        if progress_bar and i > info_window_size and i % info_window_size == 0:
            info_str = ""
            for collector in collectors:
                info_str += f"{collector.abbreviation} = {collector.get_recent_average(i, info_window_size):.2f}; "
            t.set_description(info_str[:-2], refresh=True)  # type: ignore[attr-defined]

        # abort if there are no chains left
        if pos.shape[0] <= 2:
            raise StopIteration(f"No chains left after {i} iterations. Stopping.")

        # update step(s)
        pos, touched_chain, action, success = updater(pos)

        # store observable measurements
        for collector in collectors:
            collector.collect(pos, i, touched_chain, action, success)

        # update rates
        rates[action][0] += 1
        rates[action][1] += success

    # make results available
    results = {"rates": rates}

    for collector in collectors:
        results[collector.name] = collector.get(i)
        with contextlib.suppress(AttributeError):
            results[collector.name + "_clean"] = collector.get_clean(i)
        # if collector.hasattr("get_clean"):
        #     results[collector.name+"_clean"] = collector.get_clean(i)

    return results
