# iLQR
An implementation of the Iterative Linear Quadratic Regulator (iLQR) method to control nonlinear dynamical systems.

## Description
This project was developed as part of the course "Introduction to Robotics" at the École Normale Supérieure (ENS) Paris. The report of the project can be found [here](report/report.pdf).

## Results

https://github.com/user-attachments/assets/ed4edef0-6e21-4d81-9174-b4a7326108af

https://github.com/user-attachments/assets/6624fd11-51b0-4a54-8df9-0c984046a084



## Installation
The use of [miniconda](https://docs.conda.io/en/latest/miniconda.html) is recommended to manage the dependencies. To install the dependencies, run the following command:
```bash
conda env create -f ilqr_demo_env.yml
```
To activate the environment, run:
```bash
conda activate ilqr_demo
```

This project uses [maturin](https://www.maturin.rs/) as the build system for the Rust and Python bindings. It can be installed directly using `pip`:
```bash
pip install maturin
```
To build the Rust code and install it directly as a Python package in the current `ilqr_demo` virtual environment, run:
```bash
maturin develop --release
```
You can then take a look at the code demos in the [`python/examples`](python/examples/) directory. Be sure to select the correct virtual environment before running the demos.

## Examples
The following examples are implemented in the [`python/examples`](python/examples/) directory. A checkmark indicates that the example can be successfully solved at the moment:
- [x] [Inverted Pendulum](python/examples/inverted_pendulum.ipynb): works with a number of pendulums `Nb_pendulum` equal to 1, 2
- [x] [Cartpole](python/examples/cartpole.ipynb)
