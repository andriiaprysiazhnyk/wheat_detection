import os
import yaml
import random
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from wheat_detection.model_training.datasets import WheatDataset
from wheat_detection.model_training.augmentations import get_transforms
from wheat_detection.model_training.models import get_network
from wheat_detection.model_training.trainer import Trainer


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    # read config
    with open("config.yaml") as config_file:
        config = yaml.full_load(config_file)

    # create folder for logs
    experiment_name = f"{config['model']['arch']}-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    config["log_dir"] = os.path.join(config["log_dir"], experiment_name)
    os.makedirs(config["log_dir"])

    # create summary-writer
    summary_writer = SummaryWriter(config["log_dir"])

    # copy config into logs dir
    with open(os.path.join(config["log_dir"], "config.yaml"), "w") as config_copy:
        yaml.dump(config, config_copy)

    # create data transforms
    train_transforms = get_transforms(config["train"]["transform"])
    val_transforms = get_transforms(config["val"]["transform"])

    # create data-loaders
    drop_empty_images = config["model"]["arch"] == "retina_net"
    train_ds = WheatDataset(config["train"]["path"], train_transforms, drop_empty_images)
    val_ds = WheatDataset(config["val"]["path"], train_transforms, drop_empty_images)
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)
    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=collate_fn)

    # create model
    model = get_network(config["model"])

    # configure device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # train model
    trainer = Trainer(config, train_dl, val_dl, model, summary_writer, device)
    trainer.train()
