
// Shared scalar aliases and project-wide constants for the Fashion-MNIST training pipeline.

#pragma once
#include <cstddef>
#include <cstdint>

using xfloat = float;

constexpr int EPOCHS = 10;
constexpr int MNIST_CLASSES = 10;
constexpr xfloat LEARNING_RATE = 0.1f;
constexpr xfloat MNIST_MAX_VAL = 255.0f;

