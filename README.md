# nn

A small, dependency-free Fashion-MNIST classifier in C++.

The current codebase uses a multilayer perceptron with a split design:

- `nn`: model structure and inference/training step logic
- `trainer`: epoch orchestration and metric aggregation
- `dataset_pipeline`: CSV loading and dataset validation
- Mini-batch SGD with momentum

The Release x64 build also uses AVX2 intrinsics in the dense-layer hot paths.

## Current default

Default runtime configuration:

- Topology: `784 -> 100 -> 50 -> 10`
- Hidden activation: `sigmoid`
- Output activation: `softmax`
- Learning rate: `0.04`
- Momentum: `0.9`
- Batch size: `16`
- Epochs: `10`
- Initialization: scaled Xavier-style initialization

On this machine, the current default configuration reached approximately:

- Training accuracy: `53961 / 60000`
- Evaluation accuracy: `8834 / 10000`
- Total runtime: about `13 seconds`

## Fashion-MNIST expectations

For a simple fully connected network, Fashion-MNIST test accuracy is usually in the high-80% range.

Practical expectations are roughly:

- Simple MLP: about `87%` to `89%`
- Better-tuned dense models: sometimes around `89%+`
- CNNs: typically low-90% and above

So the current result is reasonable for a small no-dependency MLP. If the goal is clearly above 90%, the architecture usually needs to move beyond a plain dense network.

## Tuning without recompiling

The binary supports runtime hyperparameter overrides via environment variables:

- `NN_HIDDEN1`
- `NN_HIDDEN2`
- `NN_LR`
- `NN_MOMENTUM`
- `NN_BATCH_SIZE`
- `NN_EPOCHS`

Example PowerShell usage:

```powershell
$env:NN_LR = '0.04'
$env:NN_MOMENTUM = '0.9'
$env:NN_BATCH_SIZE = '16'
.\bin\nn.exe
```

```powershell
$env:NN_HIDDEN1 = '120'
$env:NN_HIDDEN2 = '60'
$env:NN_LR = '0.04'
$env:NN_MOMENTUM = '0.9'
$env:NN_BATCH_SIZE = '16'
.\bin\nn.exe
```

One wider model that improved accuracy during tuning was:

- Hidden widths: `120`, `60`
- Learning rate: `0.04`
- Evaluation accuracy: about `8856 / 10000`
- Runtime: about `17 seconds`

That is a reasonable optional tradeoff, but the default remains tuned for the original runtime target.

## Build and run

Build Release x64 with MSBuild:

```powershell
& 'C:\Program Files\Microsoft Visual Studio\18\Enterprise\MSBuild\Current\Bin\amd64\MSBuild.exe' .\nn.sln /p:Configuration=Release /p:Platform=x64
```

Run from the workspace root so the dataset paths resolve:

```powershell
.\bin\nn.exe
```