
// Entry point that loads Fashion-MNIST CSV data, configures the MLP, and runs training plus evaluation.

#include <ctime>
#include <cstdlib>
#include <exception>
#include <iostream>

#include "nn.h"
#include "pipeline.h"
#include "trainer.h"

constexpr char training_data_file[] = ".\\data\\fashion-mnist_train.csv";
constexpr char evaluation_data_file[] = ".\\data\\fashion-mnist_test.csv";

namespace
{
int read_env_int(const char* name, const int fallback)
{
    if (const char* value = std::getenv(name))
    {
        char* end = nullptr;
        const long parsed = std::strtol(value, &end, 10);
        if (end != value && *end == '\0')
        {
            return static_cast<int>(parsed);
        }
    }

    return fallback;
}

xfloat read_env_float(const char* name, const xfloat fallback)
{
    if (const char* value = std::getenv(name))
    {
        char* end = nullptr;
        const float parsed = std::strtof(value, &end);
        if (end != value && *end == '\0')
        {
            return parsed;
        }
    }

    return fallback;
}
}

int main(int argc, char* argv[])
{   
    const auto start = time(nullptr);
    dataset_pipeline pipeline(MNIST_CLASSES);
    const int hidden1 = read_env_int("NN_HIDDEN1", 100);
    const int hidden2 = read_env_int("NN_HIDDEN2", 50);
    const int epochs = read_env_int("NN_EPOCHS", 10);
    const int batch_size = read_env_int("NN_BATCH_SIZE", 16);
    const xfloat learning_rate = read_env_float("NN_LR", 0.04f);
    const xfloat momentum = read_env_float("NN_MOMENTUM", 0.9f);
    const trainer_config training_config{ epochs, batch_size };
    trainer model_trainer(training_config);

    if (!pipeline.load({ training_data_file, evaluation_data_file, MNIST_MAX_VAL, true }))
    {
        return 1;
    }

    const nn_config model_config{
        { pipeline.input_dimensions(), hidden1, hidden2, MNIST_CLASSES },
        learning_rate,
        momentum,
        true,
        -1.0f,
        1.0f,
    };

    try
    {
        nn fcn;
        fcn.compile(model_config);
        fcn.summary();
        model_trainer.fit(fcn, pipeline.training_data());
        model_trainer.evaluate(fcn, pipeline.evaluation_data());
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Network initialization failed: " << ex.what() << std::endl;
        return 1;
    }

    const auto end = time(nullptr);
    std::cout << "\n\nTime taken: " << end - start << " seconds\n";

    return(0);
}
