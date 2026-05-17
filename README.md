# Studious

Studious is a minimal tensor framework built for educational purposes. It is meant to be small and readable.

The goal is to implement the core ideas behind modern tensor libraries from scratch, including automatic differentiation and execution on CPU, CUDA and Apple Metal.

## Example usage

Early work is in progress with initial focus on the CPU implementation.

```python
import studious.numpy as snp

x = snp.array([[1, 2, 3], [4, 5, 6]])
print(x)
```