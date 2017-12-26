##############################################################################
Naive CNN
##############################################################################

Main Process

::

    image processor
         |
         |
    net initializer  --> initialize weights of each layer
                     --> train   --> image_batch
                                 --> single image --> learning_rate
                                                  --> forward
                                                  --> backward


Trainer

::

    each 500-step:
        learning_rate = learning_rate * pow (0.1, exponent);
        exponent++;
        step = 0;


    forward:

    backward:


Model

::

                  input                             kernel                    feature_map
    --------------------------------------------------------------------------------------------------
     conv1      Tensor<double,32,32,3>        Tensor<double,5,5,3,64>    Tensor<double,32,32,64>

                  input                             feature_max_map           feature_map
    --------------------------------------------------------------------------------------------------
     pool       Tensor<double,32,32,64>       Tensor<bool,32,32,64>       Tensor<double,16,16,64>  

                  input                             kernel                    feature_map
    --------------------------------------------------------------------------------------------------
     conv2      Tensor<double,16,16,64>        Tensor<double,5,5,64,64>    Tensor<double,16,16,64>

                  input                             feature_max_map           feature_map
    --------------------------------------------------------------------------------------------------
     pool2      Tensor<double,16,16,64>       Tensor<bool,16,16,64>        Tensor<double,8,8,64>  

                  input                             feature_max_map           feature_map
    --------------------------------------------------------------------------------------------------
     local3     Eigen::VectorXd(4096)        Eigen::MatrixXd(384,4096)    Eigen::VectorXd(384)

Neurons

::

                  input                             kernel                    feature_map
    --------------------------------------------------------------------------------------------------
     conv1      32*32*3=3072                     5*5*64=1600                 32*32*64=65536

                  input                             pool_max_map              feature_map
    --------------------------------------------------------------------------------------------------
     pool       32*32*64=65536                   32*32*64=65536              16*16*64=16384

                  input                             kernel                    feature_map
    --------------------------------------------------------------------------------------------------
     conv2      16*16*64=16384                   5*5*64=1600                 16*16*64=16384

                  input                             kernel                    feature_map
    --------------------------------------------------------------------------------------------------
     conv2      16*16*64=16384                   16*16*64=16384              8*8*64=4096
 
