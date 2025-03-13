# DCGAN

## About

Implementation of paper '[Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)', Alec Radford, Luke Metz and Soumith Chintala.

This implementation is in `python3` using [Keras](https://keras.io/) framework with [Tensorflow](https://www.tensorflow.org/) as backend.

The model is trained on the [LUSN](http://lsun.cs.princeton.edu/2017/) dataset. 

The original dataset is in a somewhat awkward format (lmdb) and the widely-used bedroom category is very large (43GB), and it requires a python2-only script to download it. 

Therefore there is a repackaged version as a simple folder of jpgs, containing a random sample. The partial dataset with images in JPG format can be found at [LSUN bedroom scene 20% sample](https://www.kaggle.com/jhoward/lsun_bedroom/home) on Kaggle and is prepared by [Jeremy Howard](http://www.fast.ai/about/#jeremy).

When reading the images, folder arrangement of this dataset should be carefully taken care of.


## Architecture

<p align="center">
    <img src="https://github.com/manideep2510/DCGAN_LSUN/blob/master/writeup/generator.png" width="800"\>
</p>

Architecture guidelines for stable Deep Convolutional GANs,

- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

## References

[1] [Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)', Alec Radford, Luke Metz and Soumith Chintala.

[2] [Large-Scale Scene Understanding](http://lsun.cs.princeton.edu/2017/)

[3] [LSUN bedroom scene 20% sample](https://www.kaggle.com/jhoward/lsun_bedroom/home)

[4] [Keras](https://keras.io/)
