# Studious

Studious is a minimal tensor framework built for educational purposes. It is meant to be small and readable.

The goal is to implement the core ideas behind modern tensor libraries from scratch, including automatic differentiation and execution on CPU, CUDA and Apple Metal.

Early work is in progress with initial focus on the CPU implementation.

## Installation

Clone the repository:

```bash
git clone https://github.com/jvrdnd/studious.git
cd studious
```

Set up the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install nanobind
```

Build the module:

```bash
make
```

You can then use the package from the project root:

```python
import studious

print(studious)
```

To clean build artifacts:

```bash
make clean
```

## Usage

Create tensors using the NumPy API:

```python
import studious.numpy as snp

x = snp.array([[1, 2, 3], [4, 5, 6]])
print(x)
```