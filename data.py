import torch
import torchvision
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class MNISTDataset(Dataset):
    def __init__(self):
        self.dataset = torchvision.datasets.MNIST(
            root="data",
            download=True
        )

        self.dataset.data = self.dataset.data
        # [:1000]
        # [:10000]
        self.dataset.targets = self.dataset.targets
        # [:1000]
        # [:10000]
        # self.images = self.dataset.data[:1000]
        self.images = (self.dataset.data > 127).float()
        self.new_size=64
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Grab the original 28x28 binary image
        image_28 = self.images[idx]  # shape: [28, 28]

        # Upsample to new_size x new_size
        # F.interpolate needs a 4D tensor: (B, C, H, W)
        # So first unsqueeze twice, then squeeze back
        image_up = F.interpolate(
            image_28.unsqueeze(0).unsqueeze(0),  # shape [1,1,28,28]
            size=self.new_size, 
            mode='nearest'
        ).squeeze(0).squeeze(0)  # shape [new_size, new_size]

        # Now get the nonzero coordinates (more points than 28x28!)
        coords = torch.nonzero(image_up).float()

        # Normalize into [0,1] by dividing by (new_size - 1)
        coords[:, 0] /= (self.new_size - 1)
        coords[:, 1] /= (self.new_size - 1)

        return coords

    # def __getitem__(self, idx):
    #     image = self.images[idx]
    #     coords = torch.nonzero(image).float()
    #     coords[:, 0] /= 27.0
    #     coords[:, 1] /= 27.0
    #     # coords[:, 0] = coords[:, 0] / 27.0 * 2.0 - 1.0
    #     # coords[:, 1] = coords[:, 1] / 27.0 * 2.0 - 1.0

        
    #     return coords
        # return image

# dataset = MNISTDataset()
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# sample = next(iter(dataloader))
# print(sample)
# sample = sample.to("cuda:0")