#include <iostream>

#include <gtest/gtest.h>

#include "core/tensor/tensor.h"


namespace ncnn
{
// -----------------------------------------------------------------------------------------------


TEST (TensorTests, testGetDimension)
{
    Tensor<double, 5, 5, 3, 64> conv_kernel;
    ASSERT_EQ (conv_kernel.get_dimension(), 4);
}


TEST (TensorTests, testGetShape)
{
    Tensor<double, 5, 5, 3, 64> conv_kernel;
    std::vector<std::size_t> shape = conv_kernel.get_shape();
    std::vector<std::size_t> shape_correct = {5, 5, 3, 64};
    // for (auto const& value: shape)
    //     std::cout << value << std::endl;
    ASSERT_EQ (conv_kernel.get_shape(), shape_correct);
    ASSERT_EQ (conv_kernel.get_shape()[0], 5);
    ASSERT_EQ (conv_kernel.get_shape()[1], 5);
    ASSERT_EQ (conv_kernel.get_shape()[2], 3);
    ASSERT_EQ (conv_kernel.get_shape()[3], 64);
}


TEST (TensorTests, testGetSize)
{
    Tensor<double, 5, 5, 3, 64> conv_kernel;
    ASSERT_EQ (conv_kernel.get_size(), 4800);
}


TEST (TensorTests, testGetData)
{
    Tensor<double, 5, 5, 3, 64> conv_kernel_double;
    double* data_double = conv_kernel_double.get_data();
    // std::cout << *data_double << std::endl;
    ASSERT_EQ (*data_double, 4.94066e-324);

    Tensor<int, 5, 5, 3, 64> conv_kernel_int;
    int* data_int = conv_kernel_int.get_data();
    ASSERT_EQ (*data_int, 0);
}

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn


int main (int argc, char** argv)
{
    testing::InitGoogleTest (&argc, argv);
    return RUN_ALL_TESTS ();
}
