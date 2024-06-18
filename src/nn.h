#pragma once

#include <vector>
#include "common.h"

class dataset;

// Multi Layer Perceptron
class nn
{
public:
    xfloat** z, ** a, ** delta, *** weights;

    std::vector<int> layers;

    void set_layers(const std::vector<int>& l);
    void set_z(const std::vector<int>& l);
    void set_a(const std::vector<int>& l);
    void set_delta(const std::vector<int>& l);
    void set_weights(const std::vector<int>& l, const xfloat min, const xfloat max);
    void compile(const std::vector<int>& l, const xfloat min, const xfloat max);
    void zero_grad(const xfloat* X) const;
    void forward(void) const;
    void back_propagation(const xfloat* Y) const;
    void optimize(void) const;
    int get_label(const xfloat* y_pred) const;
    int predict(const xfloat* X) const;
    xfloat mse_loss(const xfloat* Y, int dim) const;
    int accuracy(const xfloat* Y, int dim) const;
    void fit(const dataset &data) const;
    void evaluate(const dataset &data);
    void summary(void) const;

    nn() = default;

    ~nn()
    {
        for (int i = 0; i < layers.size(); i += 1)
        {
            delete[] z[i];
            delete[] a[i];
        }
        delete[] z;
        delete[] a;
        for (int i = 0; i < layers.size() - 1; i += 1)
        {
            delete[] delta[i];
        }
        delete[] delta;
        for (int i = 1; i < layers.size(); i += 1)
        {
            for (int j = 0; j < layers[i] - 1; j += 1)
            {
                delete[] weights[i - 1][j];
            }
            delete[] weights[i - 1];
        }
        delete[] weights;
    }
};