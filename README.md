# avalanchepy

Python implementation of the avalanche sampling algorithm proposed in:
Herzog, M. P., von Campe, H., Kuntz, R. M., Röver, L., & Schäfer, B. M. (2023). Partition function approach to non-Gaussian likelihoods: macrocanonical partitions and replicating Markov-chains. arXiv preprint [arXiv:2311.16218](https://arxiv.org/abs/2311.16218).

- **Github repository**: <https://github.com/cosmostatistics/avalanchepy/>
- **Documentation** <https://cosmostatistics.github.io/avalanchepy/>

## Installation

### Clone from github

```bash
git clone git@github.com:cosmostatistics/avalanchepy.git
cd avalanchepy
```

### Install using poetry

```bash
make install
```

### Alternative: Install using conda

```bash
conda env create -f environment.yaml       # create conda environment
conda activate avalanchepy-env             # activate environment
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

and fire away! To start with, you might consider looking at our example in the [documentation](cosmostatistics.github.io/avalanchepy).
