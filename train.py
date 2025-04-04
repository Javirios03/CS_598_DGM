#import ml_collections
import torch
import torch.optim as optim
import pytorch_lightning as pl
import json
import os
import torch.distributed as dist
import warnings
import datetime

from data import MNISTDataset
from torch.utils.data import DataLoader
from config import create_config
from model import SetFlowModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities import rank_zero_only

torch.set_float32_matmul_precision("medium")
pl.seed_everything(42, workers=True)

config = create_config()
timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

@rank_zero_only
def setup_experiment_dir():
    os.makedirs(f"weights/{timestamp}", exist_ok=True)
    with open(f"weights/{timestamp}/config.json", "w") as f:
        f.write(config.to_json())

setup_experiment_dir()

checkpoint_callback = ModelCheckpoint(
    save_top_k=-1,
    every_n_epochs=1,
    monitor="train_loss",
    dirpath=f"weights/{timestamp}",
    filename="checkpoint-{epoch:02d}",
)

model = SetFlowModule(config=config)
dataset = MNISTDataset()
dataloader = DataLoader(dataset, batch_size=1, num_workers=8)

trainer = pl.Trainer(
    accelerator="gpu",
    devices=10,
    max_epochs=config.training.epochs,
    # gradient_clip_val=1.0,
    num_sanity_val_steps=0,
    callbacks=[checkpoint_callback]
)

trainer.fit(
    model=model, 
    train_dataloaders=dataloader,
)
