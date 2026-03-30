# Workspace Overview

- This workspace contains a small dependency-free C++ Fashion-MNIST classifier built as a Visual Studio solution in `nn.sln`.
- Core code lives in `src/`: `pipeline` loads CSV data into `dataset`, `nn` implements the multilayer perceptron, and `trainer` runs mini-batch SGD plus evaluation.
- `main.cpp` wires together runtime hyperparameters, dataset loading, model compilation, training, and evaluation.
- `py/torch-test.py` is a PyTorch baseline for comparing the C++ model against a reference implementation.
- Training and test CSV files live in `data/`, and the Release executable is expected at `bin/nn.exe`.
- Build Release x64 with MSBuild and run from the workspace root so relative dataset paths resolve correctly.