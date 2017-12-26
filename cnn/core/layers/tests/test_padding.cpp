#include <iostream>

#include <gtest/gtest.h>

#include "core/tensor/tensor.h"
#include "core/layers/padding.h"

namespace ncnn
{
// -----------------------------------------------------------------------------------------------


TEST (PaddingTests, testGetPaddingShape)
{
    Tensor<double, 32, 32, 3> input;
    Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input);
    std::vector<std::size_t> padding_shape = input_padded.get_padding_shape();
    std::vector<std::size_t> padding_shape_correct = {4, 4, 2};
    ASSERT_EQ (padding_shape, padding_shape_correct);
}


TEST (PaddingTests, testGetPaddedShape)
{
    Tensor<double, 32, 32, 3> input;
    Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input);
    std::vector<std::size_t> padded_shape = input_padded.get_padded_shape();
    std::vector<std::size_t> padded_shape_correct = {40, 40, 7};
    ASSERT_EQ (padded_shape, padded_shape_correct);
}


TEST (PaddingTests, testGetPaddedSize)
{
    Tensor<double, 32, 32, 3> input;
    Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input);
    std::size_t padded_size = input_padded.get_padded_size();
    ASSERT_EQ (padded_size, 11200);
}


// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn


int main (int argc, char** argv)
{
    testing::InitGoogleTest (&argc, argv);
    return RUN_ALL_TESTS ();
}
