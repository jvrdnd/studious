# Studious

Studious is a minimal tensor framework built for educational purposes. It is meant to be small and readable.

The goal is to implement the core ideas behind modern tensor libraries from scratch, including automatic differentiation and execution on CPU, CUDA and Apple Metal.

Work in progress.

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

Install CMake 3.15+:

```bash
# macOS
brew install cmake

# Ubuntu/Debian
sudo apt update && sudo apt install -y cmake

# Windows
winget install Kitware.CMake
```

Build the module:

```bash
make
```

Use the package from the project root:

```bash
python -c "import studious; print(studious)"
```

To clean build artifacts:

```bash
make clean
```