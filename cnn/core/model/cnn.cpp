#include <iostream>
#include <functional>

#include "core/model/cnn.h"
#include "core/layers/padding.h"
#include "core/layers/convolve.h"
#include "core/layers/pool.h"
#include "core/layers/sigmoid.h"
#include "core/layers/softmax.h"


namespace ncnn
{
// ---------------------------------------------------------------------------

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork ()
{
    // initialize the weights randomly
    std::function<void(double*, std::size_t)> initialize = [this](double* data, std::size_t size)
    {
        std::cout << "  * data: " << *data << std::endl;
        std::cout << "  * size: " << size << std::endl;
        for (double* i = data; i < data+size; ++i)
        {
            *i = normal_distributor.next ();
            // std::cout << "  * i: " << *i << std::endl;
        }
    };
 
    std::cout << "layer initializer: initializing the weights of the layer ... " << std::endl;
    std::cout << "* layer 1: conv1_kernel" << std::endl;
    initialize (conv1_kernel.get_data(), conv1_kernel.get_size());
    std::cout << "* layer 2: conv2_kernel" << std::endl;
    initialize (conv2_kernel.get_data(), conv2_kernel.get_size());
    std::cout << "* layer 3: local3" << std::endl;
    initialize (local3_weights.data(), local3_weights.size());
    initialize (local3_bias.data(), local3_bias.size());
    std::cout << "* layer 4: local4" << std::endl;
    initialize (local4_weights.data(), local4_weights.size());
    initialize (local4_bias.data(), local4_bias.size());
    std::cout << "* layer 5: softmax" << std::endl;
    initialize (softmax_weights.data(), softmax_weights.size()); 
    initialize (softmax_bias.data(), softmax_bias.size());
}


void ConvolutionalNeuralNetwork::train (ConvolutionalNeuralNetwork::image_batch& batch,
                                        ConvolutionalNeuralNetwork::label_batch& labels)
{
    double learning_rate = 0.1;
    int step = 0;
    int exponent = 2;

    Tensor<double, 32, 32, 3> image;
    int label;

    // target: 1 * 10 dimensions, value = 0
    Eigen::VectorXd target = Eigen::VectorXd::Constant (10, 0.0);

    std::size_t batchsize = batch.get_shape()[0];
    // TODO: batchsize = 1 for debugging
    // for (int i = 0; i < batchsize; ++i)
    for (int i = 0; i < 1; ++i)
    {
        // get the image address
        image.data = batch.data + i * image.get_size();
        target = Eigen::VectorXd::Constant (10, 0.0);
        label = labels (i);
        target(label) = 1.0;
    
        // let the learning rate decay over time
        if (step == 500)
        {
            learning_rate = learning_rate * pow (0.1, exponent);
            exponent++;
            step = 0;
        }

        // Eigen::MatrixXd result = forward(image);
        backward (learning_rate, forward(image), target, image);
        step++; 
    }
    
}


double ConvolutionalNeuralNetwork::test (image_batch& test_batch, label_batch& labels)
{

    std::size_t batchsize = test_batch.get_shape()[0];
    Tensor<double, 32, 32, 3> image;
    Eigen::VectorXd target;
    byte label;

    int correct = 0;
    // TODO: update batchsize
    // for (int i = 0; i < batchsize; ++i)
    for (int i = 0; i < 1; ++i)
    {
        image.data = test_batch.data + i * image.get_size();
        target = Eigen::VectorXd::Constant (0.0, 10);
        label = labels(i);
        target (label) = 1.0;

        auto cnn_output = forward (image);
        int max = cnn_output.maxCoeff();
        for (int j = 0; j < cnn_output.size(); ++j)
        {
            if (j = max) cnn_output(j) = 1.0;
            else cnn_output(j) = 0.0;
        }
        double norm = (target - cnn_output).squaredNorm();
        if ( norm < 0.001 ) correct += 1;
    }

    return correct / ((double)batchsize);
}


Eigen::MatrixXd ConvolutionalNeuralNetwork::forward (Tensor<double, 32, 32, 3>& input)
{

    // initialize pool_max_map
    std::function<void(bool*, std::size_t)> set_false = [this](bool* data, std::size_t size)
    {
        for (bool* i = data; i < data + size; ++i)
            *i = false;
    };

    // forward
    std::cout << "image forward ... " << std::endl;

    // conv1
    Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input);
    convolve (input_padded, conv1_kernel, conv1_feature_map);
    // pool1
    Padding<Tensor<double, 32, 32, 64>, 1, 1, 0> conv1_feature_map_padded(conv1_feature_map);
    set_false (pool1_max_map.get_data(), pool1_max_map.get_size());
    pool (conv1_feature_map_padded, pool1_max_map, pool1_feature_map);

    // conv2
    Padding<Tensor<double, 16, 16, 64>, 4, 4, 2> pool1_feature_map_padded(pool1_feature_map);
    convolve (pool1_feature_map_padded, conv2_kernel, conv2_feature_map);
    // pool2
    Padding<Tensor<double, 16, 16, 64>, 1, 1, 0> conv2_feature_map_padded(conv2_feature_map);
    set_false (pool2_max_map.get_data(), pool2_max_map.get_size());
    pool (conv2_feature_map_padded, pool2_max_map, pool2_feature_map);

    // local3
    // TODO: linear function (W * x + b)
    // vectorize pool2_feature_map
    Eigen::Map<Eigen::VectorXd> pool2_feature_map_vector(pool2_feature_map.get_data(), pool2_feature_map.get_size());
    // compute the sig(W * x + b)
    // Eigen::VectorXd local3_output = local3_weights * pool2_feature_map_vector + local3_bias;
    // local3_output *= -1.0;
    // local3_output = local3_output.array().exp();
    // local3_output += Eigen::VectorXd::Constant(384, 1.0);
    // local3_output.array().cwiseInverse();
    Eigen::VectorXd local3_output = local3_weights * pool2_feature_map_vector + local3_bias;
    const int local3_output_shape = 384;
    local3_output = sigmoid (pool2_feature_map_vector, local3_output, local3_output_shape);
       
    // local4
    Eigen::VectorXd local4_output = local4_weights * local3_output + local4_bias;
    const int local4_output_shape = 192;
    local4_output = sigmoid (local3_output, local4_output, local4_output_shape);

    // softmax
    // compute sfotmax(W * x + b)
    Eigen::MatrixXd cnn_output = softmax_weights * local4_output + softmax_bias;
    // Eigen::MatrixXd cnn_exp = cnn_output.array().exp();
    // double sum = cnn_exp.sum();
    // cnn_output = cnn_exp.array() / sum;
    cnn_output = softmax (cnn_output);

    return cnn_output;

}


void ConvolutionalNeuralNetwork::backward (double learning_rate,
                                           Eigen::VectorXd cnn_output,
                                           Eigen::VectorXd target,
                                           Tensor<double, 32, 32, 3>& input_image)
{
    std::cout << "(backward) backpropagate ..." << std::endl;

    // corss entropy loss
    Eigen::VectorXd v1 = Eigen::VectorXd::Constant(10, 1.0);
    double error = -target.dot(cnn_output.array().log().matrix()) - (v1 - target).dot(((v1 - cnn_output).array().log().matrix()));
    std::cout << " * error: " << error << std::endl;

    // compute the deltas for each layer
    Eigen::VectorXd softmax_bias_deltas;
    softmax_bias_deltas = - learning_rate * (cnn_output - target);
    softmax_bias += softmax_bias_deltas;

    Eigen::MatrixXd softmax_deltas;
    softmax_deltas = softmax_bias_deltas * local4_output.transpose();
    softmax_weights += softmax_deltas;

    Eigen::VectorXd local4_bias_deltas;
    local4_bias_deltas = softmax_bias_deltas.dot(softmax_weights * local4_output) * (Eigen::VectorXd::Constant(local4_output.size(), 1.0) - local4_output);
    local4_bias += local4_bias_deltas;

    Eigen::MatrixXd local4_deltas;
    local4_deltas = local4_bias_deltas * local3_output.transpose();
    local4_weights += local4_deltas;

    Eigen::VectorXd local3_bias_deltas;
    local3_bias_deltas = local4_bias_deltas.dot(local4_weights * local3_output) * (Eigen::VectorXd::Constant(local3_output.size(), 1.0) - local3_output);
    local3_bias += local3_bias_deltas;

    Eigen::MatrixXd local3_deltas;
    Eigen::Map<Eigen::VectorXd> pool2_feature_map_vector(pool2_feature_map.get_data(), pool2_feature_map.get_size());
    local3_deltas = local3_bias_deltas * pool2_feature_map_vector.transpose();
    local3_weights += local3_deltas;

    {
        auto conv2_kernel_shape = conv2_kernel.get_shape();
        const int max_col = conv2_kernel_shape[0];
        const int max_row = conv2_kernel_shape[1];
        const int max_channel = conv2_kernel_shape[2];
        const int num_filter = conv2_kernel_shape[3];

        Tensor<double, 16, 16, 64> conv2_backpropagation_feature_map;
        Tensor<double, 8, 8, 64> masking_output;
        for (int filter = 0; filter < num_filter; ++filter)
        {
            for (int col = 0; col < max_col; ++col)
            {
            for (int row = 0; row < max_row; ++row)
            {
                for (int channel = 0; channel < max_channel; ++channel)
                {
                    backconvolve (pool1_feature_map, conv2_backpropagation_feature_map, col, row, channel);
                    pool_mask (conv2_backpropagation_feature_map, pool2_max_map, masking_output);
                    Eigen::Map<Eigen::VectorXd> masking_output_vector (masking_output.get_data(), masking_output.get_size());
                    conv2_kernel (col, row, channel, filter) -= local3_bias_deltas.dot(local3_weights * masking_output_vector);
                }
            }
            }
        }
    }


    {
        auto conv1_kernel_shape = conv1_kernel.get_shape();
        const int max_col = conv1_kernel_shape[0];
        const int max_row = conv1_kernel_shape[1];
        const int max_channel = conv1_kernel_shape[2];
        const int num_filter = conv1_kernel_shape[3];

        Tensor<double, 32, 32, 64> conv1_backpropagation_feature_map;
        Tensor<double, 16, 16, 64> masking1_output;
        Tensor<double, 8, 8, 64> masking2_output;
        Tensor<double, 16, 16, 64> conv_output;

        for (int filter = 0; filter < num_filter; ++filter)
        {
            for (int col = 0; col < max_col; ++col)
            {
            for (int row = 0; row < max_row; ++row)
            {
                for (int channel = 0; channel < max_channel; ++channel)
                {
                    backconvolve (input_image, conv1_backpropagation_feature_map, col, row, channel);
                    pool_mask (conv1_backpropagation_feature_map, pool1_max_map, masking1_output);
                    convolve (masking1_output, conv2_kernel, conv_output);
                    pool_mask (conv_output, pool2_max_map, masking2_output);
                    Eigen::Map<Eigen::VectorXd> masking2_output_vector (masking2_output.get_data(), masking2_output.get_size());
                    conv1_kernel (col, row, channel, filter) -= local3_bias_deltas.dot(local3_weights * masking2_output_vector);
                }
            }
            } 
        }
    }

}
// ---------------------------------------------------------------------------
}  // namespace: ncnn
