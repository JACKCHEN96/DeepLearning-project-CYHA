# ECBM-E4040-Final
Columbia Deep-Learning course Final Project-"Multi-digit Number Recognition from Street View Imagery Paper Recurrence"

## Introduction

## Code
Currently, the directory contains five main files.

load_data: 1. Read the SVHN data (train and test); 2. Use the blue bounding box marked on the digits of street number within each individual image, find the bounding box exclosing the full street number sequence; 3. Extend the bounding box by 30% and crop the image to the size of the bounding box; 4. Resize the image to size of (54,54,3).

process_data: 1. Convert to tfrecords OR transform label??? 2. Split train data to train and validation; 3. Create some kinds of file and then output???

model: Simply build a model with eight convolution layers, two fc layers, six classifiers to predict.

train: 1. Train and tune hyper-parameter; 2. Plot.

test: 1. Use best model to test.

## Citation
- [Multi-digit Number Recognition from Street View Imagery using Deep Convolutional Neural Networks](https://arxiv.org/pdf/1312.6082.pdf)
## Contact

- [Chen Wenjie](https://github.com/JACKCHEN96)
- [Ye Hongzhe](https://github.com/hy2610)
- [Hsiung Chiaho](https://github.com/https://github.com/bearbaby1123)
