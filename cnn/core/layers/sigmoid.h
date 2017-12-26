#ifndef CNN_LAYERS_SIGMOID_H_
#define CNN_LAYERS_SIGMOID_H_

#include <Eigen/Dense>

namespace ncnn
{
// -----------------------------------------------------------------------------------------------

Eigen::VectorXd sigmoid (Eigen::VectorXd input, Eigen::VectorXd output, const int output_shape)
{
    output *= -1.0;
    output = output.array().exp();
    output += Eigen::VectorXd::Constant(output_shape, 1.0);
    return output.array().cwiseInverse();
}

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif // CNN_LAYERS_SIGMOID_H_
