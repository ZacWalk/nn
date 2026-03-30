// Activation functions and derivatives used by the multilayer perceptron.

#pragma once

#include <cmath>

#include "common.h"

__forceinline xfloat sigmoid(const xfloat x)
{
    if (x >= 0.0f)
    {
        const xfloat exponent = std::exp(-x);
        return 1.0f / (1.0f + exponent);
    }

    const xfloat exponent = std::exp(x);
    return exponent / (1.0f + exponent);
}

__forceinline constexpr xfloat sig_derivative(const xfloat x)
{
    return x * (1.0f - x);
}

constexpr xfloat relu(const xfloat x)
{
    return x > 0.0f ? x : 0.0f;
}

constexpr xfloat rel_derivative(const xfloat x)
{
    return x < 0.0f ? 0.0f : 1.0f;
}
