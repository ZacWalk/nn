
// Dataset container for normalized samples and one-hot labels loaded from CSV input.

#pragma once
#include <cstddef>
#include <vector>

#include "common.h"

class dataset
{
public:
    const int classes;
    int dimensions = 0;

    dataset(int classes) : classes(classes)
    {
    }

    void reset(int input_dimensions, std::size_t sample_capacity = 0);
    xfloat* append_sample(int label);
    void discard_last_sample(void);

    std::size_t samples() const
    {
        return sample_count;
    }

    const xfloat* sample_x(const std::size_t index) const
    {
        return X.data() + index * static_cast<std::size_t>(dimensions);
    }

    const xfloat* sample_y(const std::size_t index) const
    {
        return Y.data() + index * static_cast<std::size_t>(classes);
    }

private:
    std::size_t sample_count = 0;
    std::vector<xfloat> X;
    std::vector<xfloat> Y;
};
