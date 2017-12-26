#include "core/maths/distributions.h"

namespace ncnn
{
// -----------------------------------------------------------------------------------------------

double NormalDistribution::next ()
{
    double value = 0.;
    // value is redrawn until it does not deviate more than
    // two standard deviations from the mean 0
    do
    {
        value = normal_distribution_value (generator); 
    } while (value > 2.0 || value < -2.0);

    return value;
}


// -----------------------------------------------------------------------------------------------
}  // namespace: ncnn
