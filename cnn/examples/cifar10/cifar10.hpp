#ifndef CIFAR10_H_
#define CIFAR10_H_

#include "core/tensor/tensor.h"

typedef char byte;

const std::size_t batchsize = 10000;
const std::size_t image_height = 32;
const std::size_t image_width = 32;
const std::size_t color_channels = 3;


class CIFAR10
{
    public:
        CIFAR10 (char const *filename);

        ncnn::Tensor<double, batchsize, image_height, image_width, color_channels> images;
        ncnn::Tensor<byte, batchsize> labels; 
};


#endif  // CIFAR10_H_
