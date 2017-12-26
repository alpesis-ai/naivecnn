/*
  Tensor
 =======================

 For instance:

     Tensor<double, 32, 32, 3> image;

 ------------------------------------------------------------------------------------------------------
  Data                       Tensor<double, 32, 32, 3>
 ------------------------------------------------------------------------------------------------------
  Tensor::dimension          3
  Tensor::shape              <32, 32, 3>
  Tensor::size               32 * 32 * 3 = 3072
  Tensor::data

*/

#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include <vector>
#include <iostream>


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

template <typename datatype, int... a>
class Tensor
{
    public:
        typedef datatype datatype_t;
        datatype* data;

        // compute dimension, shape, size and allocates storage
        // - dimension: <double, 5, 5, 3, 64> -> dimension = 4
        // - size: 5 * 5 * 3 * 64 = 4800
        // - data: new datatype[size]
        Tensor ()
        {
            vectorbuilder (shape, a...);
            dimension = shape.size();
            size = 1;
            for (int i = 0; i < dimension; ++i)
                size *= shape[i];
            data = new datatype[size];
        }


        int get_dimension ()
        {
            return dimension;
        }


        std::vector<std::size_t> const& get_shape ()
        {
            return shape;
        }


        std::size_t get_size ()
        {
            return size;
        }


        datatype* get_data ()
        {
            return data;
        }    


        // returns the desired element of the tensor
        // for this function to work the number of arguments needs to be equal to
        // the dimension of the tensor
        template <typename First, typename... Args>
        datatype& operator () (First first, Args ...args)
        {
            std::size_t i = first;
            std::size_t s = 0;
            return at(i, s, args...);
        }


        // special case for one dimensional tensor
        template <typename First>
        datatype& operator () (First first)
        {
            return data[first];
        }


    protected:
        int dimension;
        std::size_t size;
        std::vector<std::size_t> shape;


        // end of the recursion of at()
        template <typename First>
        datatype& at (std::size_t i, std::size_t s, First first)
        {
            i += shape[s] * first;
            return data[i];
        }


        // only used by operator ()
        template <typename First, typename... Args>
        datatype& at (std::size_t i, std::size_t s, First first, Args ...args)
        {
            i += shape[s] * first;
            s++;
            return at<Args...>(i, s, args...);
        }

        // end of the recursion of vectorbuilder
        template <typename First>
        void vectorbuilder (std::vector<std::size_t>& v, First first)
        {
            v.push_back(first);
        }

  
        // only used in the constructor
        // It unpacks the variadic template argument and stores them in a vector.
        template <typename First, typename... Args>
        void vectorbuilder (std::vector<std::size_t>& v, First first, Args ...args)
        {
            v.push_back(first);
            vectorbuilder (v, args...);
        }
};

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif  // CNN_TENSOR_H_
