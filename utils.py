import os
from pathlib import Path
from typing import Collection, List, Union
import matplotlib.pyplot as plt
from functools import partial
from itertools import product

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

import sklearn.metrics as metrics  # pyright: ignore
from sklearn.model_selection import train_test_split  # pyright: ignore

from imageio.v2 import imread
from tqdm import tqdm, trange
from zipfile import ZipFile


#
# helper functions to load and split the data
#


def load_cifar(data_dir):
    images = []
    labels = []

    categories = os.listdir(data_dir)
    categories.sort()

    for label_id, category in tqdm(enumerate(categories), total=len(categories)):
        category_dir = os.path.join(data_dir, category)
        image_names = os.listdir(category_dir)
        for im_name in image_names:
            im_file = os.path.join(category_dir, im_name)
            images.append(np.asarray(imread(im_file)))
            labels.append(label_id)

    # from list of arrays to a single numpy array by stacking along new "batch" axis
    images = np.concatenate([im[None] for im in images], axis=0)
    labels = np.array(labels)

    return images, labels


def make_cifar_train_val_split(images, labels, validation_fraction=0.15):
    (train_images, val_images, train_labels, val_labels) = train_test_split(
        images, labels, shuffle=True, test_size=validation_fraction, stratify=labels
    )
    assert len(train_images) == len(train_labels)
    assert len(val_images) == len(val_labels)
    assert len(train_images) + len(val_images) == len(images)
    return train_images, train_labels, val_images, val_labels


def get_folder_names(zip_path: Union[Path, str]) -> List[str]:
    if isinstance(zip_path, str):
        zip_path = Path(zip_path)
    with ZipFile(zip_path, "r") as zf:
        folder_names = sorted(name for name in zf.namelist() if name.count("/") == 1)
    return folder_names


def extract(zf: ZipFile, output_path: Path, member_folders: Collection[str]):
    members = [
        name for name in zf.namelist() if name[: name.index("/") + 1] in member_folders
    ]
    zf.extractall(output_path, members)


#
# transformations and datasets
#


def to_channel_first(image, target):
    """Transform images with color channel last (WHC) to channel first (CWH)"""
    # put channel first
    image = image.transpose((2, 0, 1))
    return image, target


def normalize(image, target, channel_wise=True):
    eps = 1.0e-6
    image = image.astype("float32")
    chan_min = image.min(axis=(1, 2), keepdims=True)
    image -= chan_min
    chan_max = image.max(axis=(1, 2), keepdims=True)
    image /= chan_max + eps
    return image, target


# finally, we need to transform our input from a numpy array to a torch tensor
def to_tensor(image, target):
    return torch.from_numpy(image), torch.tensor([target], dtype=torch.int64)


# we also need a way to compose multiple transformations
def compose(image, target, transforms):
    for trafo in transforms:
        image, target = trafo(image, target)
    return image, target


class DatasetWithTransform(Dataset):
    """Our minimal dataset class. It holds data and target
    as well as optional transforms that are applied to data and target
    on the fly when data is requested via the [] operator.
    """

    def __init__(self, data, target, transform=None):
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        self.data = data
        self.target = target
        if transform is not None:
            assert callable(transform)
        self.transform = transform

    # exposes the [] operator of our class
    def __getitem__(self, index):
        data, target = self.data[index], self.target[index]
        if self.transform is not None:
            data, target = self.transform(data, target)
        return data, target

    def __len__(self):
        return self.data.shape[0]


def make_cifar_test_dataset(cifar_dir, transform=None):
    images, labels = load_cifar(os.path.join(cifar_dir, "test"))

    if transform is None:
        transform = get_default_cifar_transform()

    dataset = DatasetWithTransform(images, labels, transform=transform)
    return dataset


def make_cifar_datasets(cifar_dir, transform=None, validation_fraction=0.15):
    images, labels = load_cifar(os.path.join(cifar_dir, "train"))
    (train_images, train_labels, val_images, val_labels) = make_cifar_train_val_split(
        images, labels, validation_fraction
    )

    if transform is None:
        transform = get_default_cifar_transform()

    train_dataset = DatasetWithTransform(
        train_images, train_labels, transform=transform
    )
    val_dataset = DatasetWithTransform(val_images, val_labels, transform=transform)
    return train_dataset, val_dataset


def get_default_cifar_transform():
    trafos = [to_channel_first, normalize, to_tensor]
    trafos = partial(compose, transforms=trafos)
    return trafos


#
# visualisation functionality
#


class RunningAverage:
    """Computes and stores the average"""

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def make_confusion_matrix(labels, predictions, categories, ax):
    cm = metrics.confusion_matrix(labels, predictions)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title("Confusion matrix")
    plt.colorbar(im)
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45)
    plt.yticks(tick_marks, categories)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


### Train and Validation

from typing import Literal, assert_never


def train(
    model, 
    loader, 
    loss_function, 
    optimizer,
    device, 
    epoch,
    tb_logger, 
    log_image_interval=100,
    task: Literal["classification", "segmentation"]="classification"
):
    """ Train model for one epoch.
    
    Parameters:
    model - the model we are training
    loader - the data loader that provides the training data
        (= pairs of images and labels)
    loss_function - the loss function that will be optimized
    optimizer - the optimizer that is used to update the network parameters
        by backpropagation of the loss
    device - the device used for training. this can either be the cpu or gpu
    epoch - which trainin eppch are we in? we keep track of this for logging
    tb_logger - the tensorboard logger, it is used to communicate with tensorboard
    log_image_interval - how often do we send images to tensborboard?
    task - the type of task we are solving (classification or segmentation)
    """

    # set model to train mode
    model.train()
    
    # iterate over the training batches provided by the loader
    n_batches = len(loader)
    for batch_id, (x, y) in enumerate(loader):
       
        # send data and target tensors to the active device
        x = x.to(device)
        y = y.to(device)
        
        # set the gradients to zero, to start with "clean" gradients
        # in this training iteration
        optimizer.zero_grad()
        
        # apply the model to get the prediction
        prediction = model(x)
        
        # calculate the loss
        if task == "classification":
            # (negative log likelihood loss)
            # the loss function expects a 1d tensor, but we get a 2d tensor
            # with singleton second dimension, so we get rid of this dimension
            loss_value = loss_function(prediction, y[:, 0])
        elif task == "segmentation":
            loss_value = loss_function(prediction, y)
        else:
            assert_never(task)
        
        # calculate the gradients (`loss.backward()`) 
        # and apply them to the model parameters with
        # to our optimizer (`optimizer.step()`)
        loss_value.backward()
        optimizer.step()
        
        # log the loss value to tensorboard
        step = epoch * n_batches + batch_id
        tb_logger.add_scalar(tag='train-loss', 
                             scalar_value=loss_value.item(),
                             global_step=step)
        
        # Logging the learning rate to tensorboard
        lr = optimizer.param_groups[0]['lr']
        tb_logger.add_scalar(tag='learning-rate',
                              scalar_value=lr,
                              global_step=step)

        # check if we log images, and if we do then send the
        # current image to tensorboard
        if log_image_interval is not None and step % log_image_interval == 0:
            tb_logger.add_images(tag='input', 
                                 img_tensor=x.to('cpu'),
                                 global_step=step)
    

# the validation function takes the model, runs prediction for
# all images provided by the loader and evaluates the results.
def validate(
    model, 
    loader,
    loss_function, 
    device,
    step,
    metric,
    tb_logger=None,
    task: Literal["classification", "segmentation"]="classification"
):
    """
    Validate the model predictions.
    
    Parameters:
    model - the model to be evaluated
    loader - the loader providing images and labels
    loss_function - the loss function
    device - the device used for prediction (cpu or gpu)
    step - the current training step. we need to know this for logging
    tb_logger - the tensorboard logger. if 'None', logging is disabled
    """
    # set the model to eval mode
    model.eval()
   
    # we record the loss and the predictions / labels for all samples
    predictions = []
    labels = []

    val_scores = RunningAverage()
    val_losses = RunningAverage()
    
    # the model parameters should not be updated during validation
    # torch.no_grad disables gradient updates in its scope
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            prediction = model(x)
            
            if task == "classification":
                # update the loss 
                val_loss = loss_function(prediction, y[:, 0]).item()
                val_losses.update(val_loss, n=x.shape[0])

                # compute the most likely class predictions
                # note that 'max' returns a tuple with the 
                # index of the maximun value (which correponds to the predicted class)
                # as second entry
                prediction = prediction.max(1, keepdim=True)[1]

                # store the predictions and labels
                predictions.append(prediction[:, 0].to('cpu').numpy())
                labels.append(y[:, 0].to('cpu').numpy())

            elif task == "segmentation":
                val_loss = loss_function(prediction, y).item()
                val_losses.update(val_loss, n=x.shape[0])
                val_score = metric(prediction, y).item()
                val_scores.update(val_score, n=x.shape[0])
                
                
    if task == "classification":
        # predictions and labels to numpy arrays
        predictions = np.concatenate(predictions)
        labels = np.concatenate(labels)

        metric_value = metric(labels, predictions)
    elif task == "segmentation":
        metric_value = val_scores.avg
    
    # log the validation results if we have a tensorboard
    if tb_logger is not None:
        
        tb_logger.add_scalar(tag="validation-metric",
                             global_step=step,
                             scalar_value=metric_value)
        tb_logger.add_scalar(tag="validation-loss",
                             global_step=step,
                             scalar_value=val_losses.avg)

    # return all predictions and labels for further evaluation
    return predictions, labels, metric_value