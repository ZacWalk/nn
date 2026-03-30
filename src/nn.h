// Neural network declarations for the multilayer perceptron, its configuration, and training APIs.

#pragma once

#include <cstddef>
#include <vector>

#include "common.h"

class dataset;

struct sample_metrics
{
    xfloat loss = 0.0f;
    std::size_t correct = 0;
};

struct nn_config
{
    std::vector<int> layers;
    xfloat learning_rate = LEARNING_RATE;
    xfloat momentum = 0.9f;
    bool use_xavier_initialization = true;
    xfloat weight_min = -1.0f;
    xfloat weight_max = 1.0f;
};

// Multi Layer Perceptron
class nn
{
public:
    void compile(const nn_config& config);
    void compile(const std::vector<int>& l, const xfloat min, const xfloat max);
    int predict(const xfloat* X);
    sample_metrics train_sample(const xfloat* X, const xfloat* Y, int dim);
    void begin_batch(void);
    sample_metrics accumulate_gradients(const xfloat* X, const xfloat* Y, int dim);
    void apply_batch(std::size_t batch_size);
    sample_metrics evaluate_sample(const xfloat* X, const xfloat* Y, int dim);
    void summary(void) const;

    nn() = default;

private:
    struct layer_descriptor
    {
        int width = 0;
        int activation_width = 0;
        bool has_bias = false;
        std::size_t activation_offset = 0;
        std::size_t delta_offset = 0;
    };

    struct connection_descriptor
    {
        int input_width = 0;
        int output_width = 0;
        std::size_t weight_offset = 0;
    };


    std::vector<layer_descriptor> layers;
    std::vector<connection_descriptor> connections;
    std::vector<xfloat> activations;
    std::vector<xfloat> deltas;
    std::vector<xfloat> weights;
    std::vector<xfloat> gradient_accumulators;
    std::vector<xfloat> velocity;
    xfloat learning_rate = LEARNING_RATE;
    xfloat momentum = 0.9f;

    void load_input(const xfloat* X);
    void forward(void);
    void back_propagation(void);
    void accumulate_weight_gradients(void);
    sample_metrics evaluate_output(const xfloat* Y, int dim, bool write_output_delta);
    int get_label(const xfloat* y_pred) const;

    const layer_descriptor& output_layer(void) const;
    std::size_t parameter_count(void) const;
    xfloat* activation_ptr(std::size_t layer);
    const xfloat* activation_ptr(std::size_t layer) const;
    xfloat* delta_ptr(std::size_t layer);
    const xfloat* delta_ptr(std::size_t layer) const;
    xfloat* weight_ptr(std::size_t layer);
    const xfloat* weight_ptr(std::size_t layer) const;
    xfloat* gradient_ptr(std::size_t layer);
    xfloat* velocity_ptr(std::size_t layer);
};