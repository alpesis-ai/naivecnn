#ifndef CNN_TENSOR_TESTS_
#define CNN_TENSOR_TESTS_

#include "core/tensor/tensor.h"


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

class TensorTests
{
    protected:
        Tensor<double, 5, 5, 3, 64> conv1_kernel;
};

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif  // CNN_TENSOR_TESTS_
