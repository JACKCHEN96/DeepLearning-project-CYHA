# Neural Network and Deep Learning course projects and homeworks
Two parts:
Basic concept recurrence;
Final Project-"Multi-digit Number Recognition from Street View Imagery Paper Recurrence".

## Introduction
The course is given by Professor Zoron Kostic. This project aims to review all the basic concepts in deeplearning and implement the paper "Multi-digit Number Recognition from Street View Imagery Paper Recurrence" by Ian J. Goodfellow et al. (2014)

## Code
The repository contains several files.

DeepLearning_concept(homework): Implemented basic classifiers, MLP, dim reduction (PCA and tSNE), optimizers (SGD with Momentum, RMSprop, Adam), regularization (Dropout, Batch Normalization), data augmentation in Python (NumPy, TensorFlow and so on); Implemented convolutional neural network (used LeNet to achieve over 95% accuracy in Kaggle bottle classification
competition), recurrent neural network (compute XOR operation); realized simple translating French to English

preprocess: 1. Load training and testing dataset in the SVHN dataset; 2. Find the bounding box exclosing the full street number sequence by using the blue bounding box marked on the digits of street number within each individual image; 3. Crop the images by the enlarged box and resize them into (64, 64, 3); 4. Randomly crop the images to size of (54,54,3).

model: Build a model with eight convolution layers, two fully connected layers, six classifiers to predict.

train_1: 1. Train and tune hyper-parameter with batch size of 16; 2. Plot the accuracy over iteration.

train_2: 1. Train and tune hyper-parameter with batch size of 32; 2. Plot the accuracy over iteration.(which contributes to our best result)

test: 1. Use best model to test.

train_dct, test_dct: Training and testing with JPEG compressed data.

## Citation
- [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf)
- [SVHN Dataset](http://ufldl.stanford.edu/housenumbers/)
## Contact

- [Chen Wenjie](https://github.com/JACKCHEN96)
- [Ye Hongzhe](https://github.com/hy2610)
- [Hsiung Chiaho](https://github.com/https://github.com/bearbaby1123)

