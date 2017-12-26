#include <iostream>

#include "cifar10.hpp"
#include "core/model/cnn.h"


int main ()
{
    std::cout << "* loading the cifar-10 train images..." << std::endl;
    CIFAR10 cifar10_train_batch1 ("data/cifar10/cifar-10-batches-bin/data_batch_1.bin");
    // CIFAR10 cifar10_train_batch2 ("data/cifar10/cifar-10-batches-bin/data_batch_2.bin");
    // CIFAR10 cifar10_train_batch3 ("data/cifar10/cifar-10-batches-bin/data_batch_3.bin");
    // CIFAR10 cifar10_train_batch4 ("data/cifar10/cifar-10-batches-bin/data_batch_4.bin");
    // CIFAR10 cifar10_train_batch5 ("data/cifar10/cifar-10-batches-bin/data_batch_5.bin");

    std::cout << "* loading the cifar-10 test images..." << std::endl;
    CIFAR10 cifar10_test_batch ("data/cifar10/cifar-10-batches-bin/test_batch.bin");

    std::cout << "* initializing CNN ..." << std::endl;
    ncnn::ConvolutionalNeuralNetwork cnn;

    // train
    cnn.train (cifar10_train_batch1.images, cifar10_train_batch1.labels);    
    // test
    double accuracy = cnn.test (cifar10_test_batch.images, cifar10_test_batch.labels);
    std::cout << "accuracy: " << accuracy * 100.0 << std::endl;

    return 0;
}
