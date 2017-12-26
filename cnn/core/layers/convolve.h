/*

  Convolve
 =========================

  inputs:
    input              Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input)
    kernel             Tensor<double, 5, 5, 3, 64> conv_kernel
    feature_map        Tensor<duoble, 32, 32, 64> conv_feature_map

  feature_map (col, row, filter) = sum;
   - sum += input(col+kcol, row+krow, channel) * kernel(kcol, krow, channel, filter);

*/

#ifndef CNN_LAYERS_CONVOLVE_H_
#define CNN_LAYERS_CONVOLVE_H_

#include <iostream>


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

template <typename input_t, typename kernel_t, typename feature_map_t>
void convolve (input_t& input, kernel_t& kernel, feature_map_t& feature_map)
{
    // this implementation assumes that
    // - input: 3d
    // - kernel: 4d
    // - feature_map: 3d
    auto input_shape = input.get_shape();
    const int max_col = input_shape[0];
    const int max_row = input_shape[1];
    const int max_channel = input_shape[2];
    std::cout << " * (input) max_col: " << max_col << std::endl;
    std::cout << " * (input) max_row: " << max_row << std::endl;
    std::cout << " * (input) max_channel: " << max_channel << std::endl;

    auto kernel_shape = kernel.get_shape();
    const int max_kcol = kernel_shape[0];
    const int max_krow = kernel_shape[1];
    const int num_filter = kernel_shape[3];
    std::cout << " * (kernel) max_kcol: " << max_kcol << std::endl;
    std::cout << " * (kernel) max_krow: " << max_krow << std::endl;
    std::cout << " * (kernel) num_filter: " << num_filter << std::endl;


    double sum = 0.0;
    for (int filter = 0; filter < num_filter; ++filter)
    {
        for (int col = 0; col < max_col; ++col)
        {
        for (int row = 0; row < max_row; ++row)
        {
            sum = 0.0;
            for (int kcol = 0; kcol < max_kcol; ++kcol)
            {
            for (int krow = 0; krow < max_krow; ++krow)
            {
                for (int channel = 0; channel < max_channel; ++channel)
                {
                    sum += input(col+kcol, row+krow, channel) * kernel(kcol, krow, channel, filter);
                }
            }
            }
            // std::cout << col << ", " << row << ", " << filter << std::endl;
            // std::cout << "feature_map: " << sum << std::endl;
            feature_map(col, row, filter) = sum;
        }
        }
    }
}


/* 
   For backpropagation
   To compute the deltas of the kernel weights
*/
template <typename input_t, typename feature_map_t>
void backconvolve (input_t& input, feature_map_t& feature_map, int kcol, int krow, int channel)
{
    // This implementation assumes that
    // - input: 3d
    // - kernel: 4d
    // - feature_mamp: 3d
    auto feature_shape = feature_map.get_shape();
    const int max_col = feature_shape[0];
    const int max_row = feature_shape[1];
    const int num_filter = feature_shape[2];

    for (int filter = 0; filter < num_filter; ++filter)
    {
        for (int col = 0; col < max_col; ++col)
        {
        for (int row = 0; row < max_row; ++row)
        {
            feature_map(col, row, filter) = input(col+kcol, row+krow, channel);
        }
        }
    }
}


// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif  // CNN_LAYERS_CONVOLVE_H_
