import torch

from data import MNISTDataset
from torch.utils.data import DataLoader
from model import SetFlowModule
from config import create_config

config = create_config()
dataset = MNISTDataset()
dataloader = DataLoader(dataset, batch_size=1)

sample = next(iter(dataloader))
sample = sample.to("cuda:0")
# model = SetFlowModule(config=config).to("cuda:0")
model = SetFlowModule.load_from_checkpoint(config=config, checkpoint_path="weights/20250324-204659/checkpoint-epoch=04.ckpt")

# print(T.shape)

# print(torch.rand())

# print(sample.shape)

# model.reconstruct(sample, 10)

out = model.reconstruct(
    sample, 
    batch_size=111,
    timesteps=100
)