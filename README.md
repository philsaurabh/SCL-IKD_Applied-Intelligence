# Knowledge Distillation through Supervised Contrastive Feature Approximation
## Introduction
Knowledge distillation aims to regularize the student model
by extracting dark knowledge from a teacher network. The recent
advances in this method use self-supervised contrastive
learning alongside classification tasks to subsume more dark
knowledge from a cumbersome model. But the due to the
use of unsupervised pretraining of models, sometimes these
methods result in misclassification and performance degradation.
We propose a novel framework with improved loss
function for the knowledge distillation process that uses supervised
contrastive loss and negative cosine similarity alongside
relative cross-entropy between projection embeddings
rather than the actual logits of two models to train the student
network.
## Requirements

The code runs correctly with

* Python 3.7
* Keras 2.4.0
* TensorFlow 2.4.0

## Files

* CIFAR_10.py
* CIFAR_100.py
* TinyImagenet.py

## How to run

```bash
# GPU Id's
Set the corresponding GPU's Id's in respective codes.

# Install the basic libraries
Open the code and install all the libraries accordingly in given sequence.

# Create the CONDA Environment
Create the new conda environment ot run the files

# Run the file
All things are setup and just activate the conda environment and run python Filename.py for running the desired file.
```

### Data Preparation
All the dataset have been already imported in the corresponding codes for CIFAR10 and CIFAR100 dataset. In case of Tinyimagenet dataset instruction will be given in the file for downloading the dataset and setting the path of the dataset.
