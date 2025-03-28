import torch
import torchvision

from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self):
        self.dataset = torchvision.datasets.MNIST(
            root="data",
            download=True
        )
        
        self.images = (self.dataset.data > 127).float()
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        coords = torch.nonzero(image).float()
        coords[:, 0] /= 27.0
        coords[:, 1] /= 27.0
        # print(coords)
        
        return coords

# dataset = MNISTDataset()
# print(len(dataset))
# dataloader = DataLoader(dataset, batch_size=1)

# for batch in dataloader:
#     pass
    # print(batch.shape)
