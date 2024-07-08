"""
The most fundamental part of the sampler is given in `sampler.py` where
the main Markov loop is:
```python
for i in range(n_steps):
    state = updater(state)
    obervables[i] = [collector.collect(state) for collector in collectors]
```
From an abstract point of view, in each step, the state of the the system is
updated by an `updater`, then all possible ovservables are measured by a
list of `collectors`.

To keep the code modular, one may combine different updaters and collectors. In
particular, alternating between updates of the positions and 'kill/spawn' steps
is handled by this. Similairly, there are collectors for multiple observables
available, alternatively, one may define ones own.

All of this is wrapped by the `run` function in `interface.py` that returns
a `Result` object.
"""

__version__ = "0.1.0"

from .interface import run