#pragma once

#include "common.h"

__forceinline xfloat sigmoid(const xfloat x)
{
 //   if (x > 45.0f)
 //   {
 //       return 1.0f;
 //   }

	//if (x < -45.0f)
 //   {
	//    return 0.0f;
 //   }

    return 1.0f / (1.0f + powf(EXP, -x));
    //return 0.5f * (x / (1.0f + fabs(x)) + 1.0f);
}

__forceinline constexpr xfloat sig_derivative(const xfloat x)
{
    return (x * (1.0f - x));                 /// Sigmoid derivative formula
}

constexpr xfloat relu(const xfloat x)
{
    return (x > 0.0f ? x : 0.0f);             /// ReLU formula
}

constexpr xfloat rel_derivative(const xfloat x)
{
    return (x < 0.0f ? 0.0f : 1.0f);           /// ReLU derivative formula
}
