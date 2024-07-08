# avalanchepy

[![Release](https://img.shields.io/github/v/release/maxiherzog/avalanchepy)](https://img.shields.io/github/v/release/maxiherzog/avalanchepy)
[![Build status](https://img.shields.io/github/actions/workflow/status/maxiherzog/avalanchepy/main.yml?branch=main)](https://github.com/maxiherzog/avalanchepy/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/maxiherzog/avalanchepy)](https://img.shields.io/github/commit-activity/m/maxiherzog/avalanchepy)
[![License](https://img.shields.io/github/license/maxiherzog/avalanchepy)](https://img.shields.io/github/license/maxiherzog/avalanchepy)

Python implementation of the avalanche sampling algorithm proposed in:

Herzog, M. P., von Campe, H., Kuntz, R. M., Röver, L., & Schäfer, B. M. (2023). Partition function approach to non-Gaussian likelihoods: macrocanonical partitions and replicating Markov-chains. arXiv preprint [arXiv:2311.16218](https://arxiv.org/abs/2311.16218).

# Getting started

A simple example might look like this:

```python
import numpy as np
import avalanchepy

# Define a simple gaussian likelihood function
def loglike(theta):
    return -0.5 * np.sum(theta**2)

# Run avalanchepy
result = avalanchepy.run(loglike, mu=3, n_total_steps=30000)
result.print_summary()
```
