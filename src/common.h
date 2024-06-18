
/**
 * common.hpp
 *
 * In this header file, we define the constants
 * used throughout the project. We also
 * include all the header files necessary to
 * make the implementation work.
 */

#pragma once
#include <cstdint>


typedef intptr_t ssize_t;
typedef float xfloat;

constexpr int EPOCHS = 10;
constexpr int MNIST_CLASSES = 10;
constexpr xfloat LEARNING_RATE = 0.1;
constexpr xfloat EXP = 2.718282;
constexpr xfloat MNIST_MAX_VAL = 255.0;


constexpr int MNIST_TRAIN = 60000;
constexpr int MNIST_TEST = 10000;

