# avalanchepy

[![Release](https://img.shields.io/github/v/release/maxiherzog/avalanchepy)](https://img.shields.io/github/v/release/maxiherzog/avalanchepy)
[![Build status](https://img.shields.io/github/actions/workflow/status/maxiherzog/avalanchepy/main.yml?branch=main)](https://github.com/maxiherzog/avalanchepy/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/maxiherzog/avalanchepy/branch/main/graph/badge.svg)](https://codecov.io/gh/maxiherzog/avalanchepy)
[![Commit activity](https://img.shields.io/github/commit-activity/m/maxiherzog/avalanchepy)](https://img.shields.io/github/commit-activity/m/maxiherzog/avalanchepy)
[![License](https://img.shields.io/github/license/maxiherzog/avalanchepy)](https://img.shields.io/github/license/maxiherzog/avalanchepy)

Python implementation of the avalanche sampling algorithm proposed in:
Herzog, M. P., von Campe, H., Kuntz, R. M., Röver, L., & Schäfer, B. M. (2023). Partition function approach to non-Gaussian likelihoods: macrocanonical partitions and replicating Markov-chains. arXiv preprint [arXiv:2311.16218](https://arxiv.org/abs/2311.16218).

- **Github repository**: <https://github.com/maxiherzog/avalanchepy/>
- **Documentation** <https://maxiherzog.github.io/avalanchepy/>

## Installation

### Clone from github

```bash
git clone git@github.com:maxiherzog/avalanchepy.git
cd avalanchepy
```

### Install using poetry

```bash
make install
```

### Alternative: Install using conda

```bash
micromamba env create -f environment.yaml  # create mamba environment
micromamba activate avalanchepy-env        # activate environment
pip install -e .                           # install as an editable package
```

### Alternative: Install using pip

```bash
pip install -r requirements.txt -e .       # install requirements and package
```

## Usage

Simply use

```python
import avalanchepy
```

and fire away! To start with, you might consider looking at an [example](???) in the [documentation](???).

@maxiherzog: If we do not plan to host the docs, one might add:

## Documentation

To view the documentation,

```bash
make docs
```

and navigate to [http://127.0.0.1:8000/avalanchepy/](http://127.0.0.1:8000/avalanchepy/) in your browser.

---

Repository initiated with [fpgmaas/cookiecutter-poetry](https://github.com/fpgmaas/cookiecutter-poetry).
