#include <fstream>
#include <iostream>

#include "cifar10.hpp"


CIFAR10::CIFAR10 (char const *filename)
{
    // label
    std::size_t labelsize = 1;
    byte *label = new byte[labelsize];
    // image
    std::size_t imagesize = image_height * image_width * color_channels;
    byte *image = new byte[imagesize];

    // labels & images
    std::ifstream ifs (filename, std::ios::binary);
    for (std::size_t i = 0; i < batchsize; ++i)
    {
        // read label from data
        ifs.read(label, labelsize);
        labels(i) = *label;
        // read image from data
        ifs.read(image, imagesize);
        for (std::size_t row = 0; row < image_height; ++row)
        {
            for (std::size_t col = 0; col < image_width; ++col)
            {
                std::size_t pixel = image_height * row + col;
                // read and store all three color channels
                images(i, col, row, 0) = (double)(unsigned char)image[pixel + 0];
                images(i, col, row, 1) = (double)(unsigned char)image[pixel + image_height * image_width];
                images(i, col, row, 2) = (double)(unsigned char)image[pixel + 2 * image_height * image_width];
            }
        }
    }
}
