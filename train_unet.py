import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn as nn
from utils import (
    run_nuclei_training,
    UNet, 
    DiceCoefficient,
    NucleiDataset,
    RandomCrop,
)
import typer
from typing import Annotated


def main(
    name: Annotated[str, typer.Option(help="Name of the training run")] = "unet",
    epochs: Annotated[int, typer.Option(help="Number of epochs to train the model for")] = 10,
    train_data_path: Annotated[str, typer.Option(help="Path to the training data")] = "/scratch/talks/data/nuclei_train_data",
    val_data_path: Annotated[str, typer.Option(help="Path to the validation data")] = "/scratch/talks/data/nuclei_val_data",
):

    have_gpu = torch.cuda.is_available()
    # we need to define the device for torch, yadda yadda
    if have_gpu:
        print("GPU is available")
        device = torch.device('cuda')
    else:
        print("GPU is not available, training will run on the CPU")
        device = torch.device('cpu')

    unet_model = UNet(in_channels=1, out_channels=1, final_activation=nn.Sigmoid())
    unet_model.to(device)

    unet_optimizer = Adam(unet_model.parameters(), lr=1e-3)
    metric = DiceCoefficient()
    loss_function = nn.BCELoss()

    from torch.utils.data import DataLoader

    train_data = NucleiDataset(train_data_path, RandomCrop(256))
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, num_workers=8)

    val_data = NucleiDataset(val_data_path, RandomCrop(256))
    val_loader = DataLoader(val_data, batch_size=32, num_workers=8)

    _ = run_nuclei_training(
        model=unet_model,
        optimizer=unet_optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_function=loss_function,
        metric=metric,
        device=device,
        name=name,
        n_epochs=epochs,
    )

if __name__ == "__main__":
    typer.run(main)