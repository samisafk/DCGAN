# DCGAN with CelebA Dataset

Implementation of Deep Convolutional Generative Adversarial Networks (DCGAN) using PyTorch on the CelebA dataset.

## Paper Reference

- '[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)' by Alec Radford, Luke Metz, and Soumith Chintala.

## Dataset

- The model is trained on the [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) dataset, which contains images of celebrities.
- Images are preprocessed and stored in JPG format within the directory `./data/celeba`.

## Frameworks and Dependencies

This implementation is written in Python using the following libraries:

- [PyTorch](https://pytorch.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Matplotlib](https://matplotlib.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/)

## Model Architecture

This DCGAN follows the best practices outlined in the referenced paper:

- **Discriminator**:
  - Convolutional layers with stride instead of pooling.
  - Batch normalization layers for stability.
  - Leaky ReLU activation is used for non-linearity.
  - Final layer produces a single scalar output.

## Training Configuration

- **Dataset**: CelebA dataset, loaded using `torchvision.datasets.ImageFolder`
- **Image Size**: 64x64
- **Batch Size**: 128
- **Learning Rate**: 0.0002
- **Beta1 for Adam Optimizer**: 0.5
- **Number of Epochs**: 25

## Training Details

- The network is trained on the CelebA dataset using **PyTorch's DataLoader**
- The discriminator is optimized using both real and fake images.
- The generator is optimized to generate more realistic images.
- Training is performed for **25 epochs**.
- Fake images are generated at the end of each epoch and saved as output.
- The **discriminator and generator loss** are monitored throughout training.

## Training Process

1. Load the **CelebA dataset** from `./data/celeba` and apply preprocessing steps such as resizing, normalizing, etc.
2. **Initialize the models** with normal distribution weights.
3. **Train the models**:
   - Update discriminator using real and fake images.
   - Update generator to produce more realistic images.
   - Use **Binary Cross Entropy loss** for both models.
4. **Generate and save images** at the end of each epoch using PyTorch's `torchvision.utils.save_image` function.

## Dependencies

Make sure to install the necessary libraries:

```bash
pip install torch torchvision matplotlib pillow
```

## Usage

To train the model, run the following command:

```bash
python dcgan.py
```

## Acknowledgments

- **Paper**: '[Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks](https://arxiv.org/pdf/1511.06434.pdf)' by Alec Radford, Luke Metz, and Soumith Chint
- **Dataset**: [CelebA](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
- **PyTorch Tutorial on DCGAN**: [PyTorch DCGAN](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)

