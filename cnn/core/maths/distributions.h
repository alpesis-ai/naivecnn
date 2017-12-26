#ifndef CNN_MATHS_DISTRIBUTIONS_
#define CNN_MATHS_DISTRIBUTIONS_

#include <random>


namespace ncnn
{
// -----------------------------------------------------------------------------------------------

class NormalDistribution
{
    public:
        std::random_device random_device;
        std::mt19937 generator;
        std::normal_distribution<> normal_distribution_value;

        NormalDistribution ():generator(random_device()) {};
        double next ();
};

// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn

#endif  // CNN_MATHS_DISTRIBUTIONS
