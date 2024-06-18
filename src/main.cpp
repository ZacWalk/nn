
#include <iostream>
#include <vector>

#include "nn.h"
#include "dataset.h"

constexpr char training_data_file[] = ".\\data\\fashion-mnist_train.csv";
constexpr char evaluation_data_file[] = ".\\data\\fashion-mnist_test.csv";

int main(int argc, char* argv[])
{   
    dataset TRAIN(MNIST_CLASSES);
    dataset TEST(MNIST_CLASSES);
    const auto start = time(nullptr);

    TRAIN.read_csv(training_data_file, 0, MNIST_MAX_VAL);
    TEST.read_csv(evaluation_data_file, 1, MNIST_MAX_VAL);

    std::vector<int> vec{ TRAIN.dimensions, 100, 50, 10 }; // 8778 out of 10000 (30 sec)
    //std::vector<int> vec{ TRAIN.dimensions, 150, 50, 10 }; // 8721 out of 10000 (42 sec)
    // std::vector<int> vec{ TRAIN.dimensions, 200, 100, 50, 10 }; // 8733 out of 10000 (126 sec)
    // std::vector<int> vec { TRAIN.dimensions, 200, 10 }; // 8665 out of 10000 (47 sec)

    nn fcn;
    fcn.compile(vec, -1.0, 1.0);
    fcn.summary();
    fcn.fit(TRAIN);
    fcn.evaluate(TEST);

    const auto end = time(nullptr);
    std::cout << "\n\nTime taken: " << end - start << " seconds\n";

    return(0);
}
