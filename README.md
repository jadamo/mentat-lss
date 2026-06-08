[![Static Badge](https://img.shields.io/badge/arXiv-2603.16003-darkred)](https://arxiv.org/abs/2603.16003)

[![pytests](https://github.com/jadamo/mentat-lss/actions/workflows/pytest.yaml/badge.svg)](https://github.com/jadamo/mentat-lss/actions/workflows/pytest.yaml)
[![codecov](https://codecov.io/gh/jadamo/mentat-lss/graph/badge.svg)](https://codecov.io/gh/jadamo/mentat-lss)
![Read the Docs](https://img.shields.io/readthedocs/spherex_emu)

# MENTAT-LSS

![Realistic rendition of mentat-lss (aka spherex_emu), showing an emu inside the SPHEREx satellite dish. Image credit: Grace Gibbins](/docs/spherex_emu_logo.png)

The **M**ultipole **E**mulator for **N**onlinear **T**racer **A**nalysis of **T**wo-point statistics and **L**arge **S**cale **S**tructure (also known as spherex_emu) is a package providing tools to create and use a neural network emulator that outputs redshift-space galaxy power spectrum multipoles given a set of input cosmology + galaxy bias parameters. Said emulator is able to generate multipoles for multiple tracer and redhshift bins simultaniously. While originally designed for use in SPHEREx likelihood inference studies, mentat-lss can be used for any galaxy clustering survey (BOSS, DESI, etc).

For more details on how to use this package, check out our documentation on [ReadTheDocs](https://spherex-emu.readthedocs.io/en/latest/index.html)!

## Installing the code

This package works on both Linux and MacOS (intel and arm64) platforms, and has so-far been tested using **Python 3.9 - 3.11**. There are two main methods to install the code.

### Preliminaries

1. To enable GPU functionality for network training, make sure you have CUDA installed (or python 3.8+ if using apple silicon).
2. You will need some way to generate galaxy power spectrum multipoles to generate training sets. One option is to download and install both [ps_1loop](https://github.com/archaeo-pteryx/ps_1loop) and [ps_theory_calculator](https://github.com/archaeo-pteryx/ps_theory_calculator). You might need to request access to those repositories, in which case you can contact Yosuke Kobayashi (yosukekobayashi@arizona.edu). We are also working on interfacing this library with [cocoa](https://github.com/CosmoLike/cocoa), so stay tuned!

### From pip (recommended)

In a clean enviornment, simply run,

`pip install mentat-lss`

Alternatively, if you would like to install from source (for example, you want to add to the package, or would like
easier access to the provided config files)), you can do so in two different ways.

### From source (automatic)

1. Download this repository to your location of choice.
2. In the base directory, simply run `install.sh` in the terminal. This script will create a new anaconda enviornment, fetch the corresponding version of PyTorch, and install the code, all automatically.

### From source (manual)

1. Download this repository to your location of choice.
2. install the corresponding [PyTorch version](https://pytorch.org/get-started/locally/). If your machine doesn't have a GPU, you can skip this step.
3. In the base directory, run `python -m pip install .`, which should install this repository as a package and all required dependencies.

To run the provided unit-tests, you can run the following command in the base repo directory,

`python -m pytest tests`

## Using MENTAT-LSS

Check out our [ReadTheDocs Page](https://spherex-emu.readthedocs.io/en/latest/workflow.html) on a typical workflow process.

If you use this package for your research, please cite the following papers:

- Adamo et al (2026) - [https://arxiv.org/abs/2603.16003](https://arxiv.org/abs/2603.16003)
- symbolic_pofk ([Bartlett et al (2024)](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A.209B/abstract))

## Authors

- Joe Adamo (primary) (jadamo@arizona.edu)
- Annie Moore
- Grace Gibbins
