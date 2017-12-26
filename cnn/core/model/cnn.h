#ifndef CNN_CNN_H_
#define CNN_CNN_H_

#include <Eigen/Dense>

#include "core/tensor/tensor.h"
#include "core/maths/distributions.h"


typedef char byte;

namespace ncnn
{
// -----------------------------------------------------------------------------------------------

class ConvolutionalNeuralNetwork
{

    public:
        typedef Tensor<double, 10000, 32, 32, 3> image_batch;
        typedef Tensor<byte, 10000> label_batch;

        ConvolutionalNeuralNetwork ();

        void train (image_batch& batch, label_batch& labels);
        double test (image_batch& test_batch, label_batch& labels);

    protected:
        NormalDistribution normal_distributor;
        Eigen::MatrixXd forward (Tensor<double, 32, 32, 3>& input);
        void backward (double learning_rate, Eigen::VectorXd cnn_output, Eigen::VectorXd target, Tensor<double, 32, 32, 3>& input_image);

        // weights of each layer
        // conv1
        Tensor<double, 5, 5, 3, 64> conv1_kernel;
        Tensor<double, 32, 32, 64> conv1_feature_map;
        // pool1
        Tensor<double, 16, 16, 64> pool1_feature_map;
        Tensor<bool, 32, 32, 64> pool1_max_map;
        // conv2
        Tensor<double, 5, 5, 64, 64> conv2_kernel;
        Tensor<double, 16, 16, 64> conv2_feature_map;
        // pool2
        Tensor<double, 8, 8, 64> pool2_feature_map;
        Tensor<bool, 16, 16, 64> pool2_max_map;
        // local3: weights [4096, 384] (the output vectorized pool2: 4096 = 8 * 8 * 64)
        Eigen::MatrixXd local3_weights = Eigen::MatrixXd (384, 4096);
        Eigen::VectorXd local3_bias = Eigen::VectorXd (384);
        Eigen::VectorXd local3_output = Eigen::VectorXd (384);
        // local4
        Eigen::MatrixXd local4_weights = Eigen::MatrixXd (192, 384);
        Eigen::VectorXd local4_bias = Eigen::VectorXd (192);
        Eigen::VectorXd local4_output = Eigen::VectorXd (192);
        // softmax
        Eigen::MatrixXd softmax_weights = Eigen::MatrixXd (10, 192);
        Eigen::VectorXd softmax_bias = Eigen::VectorXd (10);
};

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif // CNN_CNN_H_ 
