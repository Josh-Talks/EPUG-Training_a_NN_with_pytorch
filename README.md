# EPUG -- Training your own neural network for image analysis with PyTorch

This repo gives a short introduction to training a neural network (NN) using PyTorch. The two notebooks,

- EPUG_train_NN.ipynb
- U-Net_extension.ipynb

introduce the basic components for working with imaging data and training your own NN. The first notebook `EPUG_train_NN_ipynb` walks through working with a dataset using Pytorch `Dataset`s and `Dataloader`s, defining your own neural network, defining a training and validation loop, training your own model and evaluating the results. The second notebook extends the lessons learned from notebook 1 to build your own U-Net based architecture and train and evaulate your model on nuclei semantic segmentation.


## Running the notebooks

The notebooks can either be run in google Collab by clicking on the links at the top of the notebooks, or they can be run locally by cloning the git repo and setting up the appropriate conda environment.

### Cloning the repo for running locally
Navigate to your desired directory then in the terminal enter the following,

`git clone https://github.com/Josh-Talks/EPUG-Training_a_NN_with_pytorch.git`


### Setting up Conda environment
The repo has an environment.yaml file that defines the neccessary dependancies. To set up your own conda environment first install conda, for example by following the installation instructions given by [miniforge](https://github.com/conda-forge/miniforge). Then navigate to the base directory of the repo and enter the following command,

`conda env create -f environment.yaml`



## `train_unet.py` script

The repo also contains a python file `train_unet.py` which can be used to train the U-Net architecture defined in the `U-Net_extension.ipynb` notebook outside of said notebook.
