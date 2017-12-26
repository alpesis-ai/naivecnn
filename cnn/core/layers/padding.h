/*
  Padding
 =============


  For instance:
        Padding<Tensor<double, 32, 32, 3>, 4, 4, 2> input_padded(input)


                                 input                     padding                 padded
  ---------------------------------------------------------------------------------------------------
    Data                  Tensor<double,32,32,3>       <Tensor<>,4,4,2>        <Tensor<>,40,40,7>
  ---------------------------------------------------------------------------------------------------
    Tensor::dimension     3                                
    Tensor::shape         <32,32,3>                        <4,4,2>                <40,40,7>
    Tensor::size          32*32*3=3072                                            40*40*7=11200
    Tensor::data

*/

#ifndef CNN_LAYERS_PADDING_H_
#define CNN_LAYERS_PADDING_H_

#include <vector>

#include "core/tensor/tensor.h"


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

template <typename tensor, int... p>
class Padding : public tensor
{
    public:
        Padding () = delete;
        Padding (tensor t)
        {
            tensor::vectorbuilder (padding_shape, p...);
            tensor::dimension = t.get_dimension ();
            tensor::shape = t.get_shape ();
            tensor::data = t.get_data ();

            for (int i = 0; i < tensor::dimension; ++i)
            {
                padded_shape.push_back (tensor::shape[i] + 2 * padding_shape[i]);
            }

            tensor::size = 1;
            for (int i = 0; i < tensor::dimension; ++i)
            {
                tensor::size *= tensor::shape[i];
            }

            padded_size = 1;
            for (int i = 0; i < tensor::dimension; ++i)
            {
                padded_size *= padded_shape[i];
            }
        }


        std::vector<std::size_t> const& get_padding_shape ()
        {
            return padding_shape;
        }

    
        std::vector<std::size_t> const& get_padded_shape ()
        {
            return padded_shape;
        }


        std::size_t get_padded_size ()
        {
            return padded_size;
        }


        // end of the recureion of check_padded()
        template <typename First>
        bool check_padded (int argnum, First first)
        {
            if ( first < padding_shape[argnum] ) return true;
            if ( first >= padding_shape[argnum] + tensor::shape[argnum] ) return true;
            return false;
        }


        // only used by operator ()
        template <typename First, typename... Args>
        bool check_padded (int argnum, First first, Args ...args)
        {
            if ( first < padding_shape[argnum] ) return true;
            if ( first >= padding_shape[argnum] + tensor::shape[argnum] ) return true;
            return check_padded(++argnum, args...);
        }


        template <typename First>
        typename tensor::datatype_t& at (std::size_t i, std::size_t s, First first)
        {
            i += tensor::shape[s-1] * ( first - padding_shape[s] );
            return tensor::data[i];
        }


        template <typename First, typename... Args>
        typename tensor::datatype_t & at (std::size_t i, std::size_t s, First first, Args ...args)
        {
            i += tensor::shape[s-1] * (first - padding_shape[s]);
            s++;
            return at<Args...>(i, s, args...);
        }


        // Checks if the indices of the desired element is inside the stored tensor.
        // If not, it returns 0.
        template <typename First, typename... Args>
        typename tensor::datatype_t& operator () (First first, Args ...args)
        {
            if (check_padded (0, first, args...))
                return return_value;

            std::size_t i = first - padding_shape[0];
            std::size_t s = 1;
            return at<Args...>(i, s, args...);
        }


        // special case for one dimensional tensor
        template <typename First>
        typename tensor::datatype_t& operator () (First first)
        {
            if (check_padded (0, first))
                return return_value;
            return tensor::data[first];
        }


    protected:
        std::size_t padded_size;
        std::vector<std::size_t> padded_shape;
        std::vector<std::size_t> padding_shape;
        typename tensor::datatype_t return_value = 0.0;
};

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif  // CNN_LAYERS_PADDING_H_
