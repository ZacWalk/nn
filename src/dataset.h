
#pragma once
#include <vector>

#include "common.h"

class dataset
{
public:
    const int classes;
    int dimensions = 0;
    std::vector<xfloat*> X;
    std::vector<xfloat*> Y;

    void read_csv(const char* filename, int dataset_flag, xfloat x_max);

    dataset(int classes) : classes(classes)
    {
    }

    ~dataset()
    {
        for (size_t i = 0u; i < X.size(); i += 1)
        {
            delete[] X[i];
            delete[] Y[i];
        }
    }

    size_t samples() const
    {
        return X.size();
    }
};
