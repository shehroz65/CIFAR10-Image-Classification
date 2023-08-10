# CIFAR10-Image-Classification

A simple CNN architecture designed to classify images from the CIFAR 10 Dataset. 

The architecture is as follows:

**Layer 1**: Convolutional layer with 3 input channels (for RGB), 10 output channels, and a kernel size of 7x7.

**Pooling Layer 1**: Max pooling layer with a size of 2x2 and a stride of 1.

**Layer 2**: Convolutional layer with 10 input channels, 21 output channels, and a kernel size of 5x5.

**Pooling Layer 2**: Max pooling layer with a size of 4x4 and a stride of 2.

**Fully Connected Layer**: A linear layer that maps from 1701 features to 10 classes.
