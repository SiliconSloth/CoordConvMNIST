# CoordConv Autoencoder for Position Detection with MNIST Digits

### Blog post: [https://siliconsloth.com/posts/autoencoders-for-position-detection/](https://siliconsloth.com/posts/autoencoders-for-position-detection/)

Implementation of an autoencoder that uses CoordConv layers to enhance extraction
of positional information from images of randomly positioned MNIST digits.

Running `main.py` trains an autoencoder with CoordConv layers and one without
CoordConv layers, then generates a figure comparing the two.

This project is explained in more detail in the corresponding blog post at [siliconsloth.com](https://siliconsloth.com/posts/autoencoders-for-position-detection/).

## Acknowledgements

The file `coord.py` was copied from [keras-coordconv](https://github.com/titu1994/keras-coordconv) by Somshubra Majumdar. The license for that repository is included in the file.

CoordConv layers were introduced by the paper [An Intriguing Failing of Convolutional Neural Networks and the CoordConv Solution](https://arxiv.org/abs/1807.03247)
by Liu et al.
