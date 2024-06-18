# nn
A simple multilevel perceptron that can train and run on the fashion mnist dataset. Implemented in C++ with no dependencies.

I wanted to experiment with the accuracy when using 8-bit log weights. Currently only mplementing with floats.

Current output:

```
Neural Network Summary:         [f := Sigmoid]

Layer 1  784 neurons
Layer 2  100 neurons
Layer 3   50 neurons
Layer 4   10 neurons

[EPOCH    1] [LOSS 0.14729] [ACCURACY  47787 out of 60000]
[EPOCH    2] [LOSS 0.11223] [ACCURACY  50813 out of 60000]
[EPOCH    3] [LOSS 0.10208] [ACCURACY  51744 out of 60000]
[EPOCH    4] [LOSS 0.09830] [ACCURACY  51982 out of 60000]
[EPOCH    5] [LOSS 0.09405] [ACCURACY  52323 out of 60000]
[EPOCH    6] [LOSS 0.09002] [ACCURACY  52710 out of 60000]
[EPOCH    7] [LOSS 0.08490] [ACCURACY  53088 out of 60000]
[EPOCH    8] [LOSS 0.08296] [ACCURACY  53326 out of 60000]
[EPOCH    9] [LOSS 0.08291] [ACCURACY  53331 out of 60000]
[EPOCH   10] [LOSS 0.08029] [ACCURACY  53493 out of 60000]

[EVALUATION] [LOSS 0.09102] [ACCURACY   8736 out of 10000]

Time taken: 15 seconds
```
