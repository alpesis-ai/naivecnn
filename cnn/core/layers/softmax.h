#ifndef CNN_LAYERS_SOFTMAX_H_
#define CNN_LAYERS_SOFTMAX_H_

#include <Eigen/Dense>


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

Eigen::MatrixXd softmax (Eigen::MatrixXd input)
{
    Eigen::MatrixXd output_exp = input.array().exp();
    double sum = output_exp.sum();
    input = output_exp.array() / sum;

    return input;
}

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif // CNN_LAYERS_SOFTMAX_H_
