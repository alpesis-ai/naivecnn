/*

  Pooling
 ======================

    For instance:

                        input                 pool_max_map                      feature_map
    ---------------------------------------------------------------------------------------------------
      pool        Tensor<double,32,32,64>   Tensor<double,32,32,64>          Tensor<bool,16,16,64>


*/

#ifndef CNN_LAYERS_POOL_H_
#define CNN_LAYERS_POOL_H_

namespace ncnn
{
// -----------------------------------------------------------------------------------------------

template <typename input_t, typename pool_max_map_t, typename feature_map_t>
void pool (input_t& input, feature_map_t& feature_map, pool_max_map_t& pool_max_map)
{
    // this implementation assumes that
    // - input: 3d
    // - feature_map: 3d
    auto input_shape = input.get_shape ();
    const int max_col = input_shape[0];
    const int max_row = input_shape[1];
    const int num_features = input_shape[2];
    const int stride = 2;

    const int max_wcol = 3;
    const int max_wrow = 3;

    for (int feature = 0; feature < num_features; ++feature)
    {
        for (int col = 0; col < max_col; col += stride)
        {
        for (int row = 0; row < max_row; row += stride)
        {
            double max = 0.0;
            for (int wcol = 0; wcol < max_wcol; ++wcol)
            {
            for (int wrow = 0; wrow < max_wrow; ++wrow)
            {
                double value = input(col+wcol, row+wrow, feature);
                if (value > max)
                {
                    max = value;
                    pool_max_map(col+wcol, row+wrow, feature) = 1;
                }
            }
            }
            // std::cout << "max: " << max << std::endl; 
            feature_map(col/stride, row/stride, feature) = max;
        }
        }
    }
}


/*
   For backpropagation
   To reconstruct the transformation of the pooling layer on the previous layer
*/
template <typename input_t, typename pool_max_map_t, typename feature_map_t>
void pool_mask (input_t& input, feature_map_t& feature_map, pool_max_map_t& pool_max_map)
{
    // This implementation assumes that
    // - input: 3d
    // - feature_map: 3d
    auto input_shape = input.get_shape();
    const int max_col = input_shape[0];
    const int max_row = input_shape[1];
    const int num_features = input_shape[2];
    const int stride = 2;

    const int max_wcol = 3;
    const int max_wrow = 3;

    for (int feature = 0; feature < num_features; ++feature)
    {
        for (int col = 0; col < max_col; col += stride)
        {
        for (int row = 0; row < max_row; row += stride)
        {
            double max = 0.0;
            for (int wcol = 0; wcol < max_wcol; ++wcol)
            {
            for (int wrow = 0; wrow < max_wrow; ++wrow)
            {
                if (pool_max_map(col+wcol, row+wrow, feature))
                {
                    max = input(col+wcol, row+wrow, feature);
                    break;
                }
            }
            }
            feature_map(col/stride, row/stride, feature) = max;
        }
        }
    }
}

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif // CNN_LAYERS_POOL_H_
