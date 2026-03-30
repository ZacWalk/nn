// CSV loading pipeline that builds validated training and evaluation datasets for the network.

#pragma once

#include "common.h"
#include "dataset.h"

struct pipeline_config
{
    const char* training_file = nullptr;
    const char* evaluation_file = nullptr;
    xfloat x_max = MNIST_MAX_VAL;
    bool has_header = true;
};

class dataset_pipeline
{
public:
    explicit dataset_pipeline(int classes);

    bool load(const pipeline_config& config);
    int input_dimensions(void) const;
    const dataset& training_data(void) const;
    const dataset& evaluation_data(void) const;

private:
    bool load_csv(dataset& target, const char* filename, xfloat x_max, bool has_header) const;
    bool validate_pair(void) const;

    dataset training;
    dataset evaluation;
};