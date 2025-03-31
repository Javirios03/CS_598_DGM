import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

# visualization
import matplotlib.pyplot as plt

from matplotlib import cm
from munkres import Munkres

# To avoide meshgrid warning
import warnings

from hungarian_algorithm import algorithm

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

class Flow(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        self.decoder = decoder
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=self.decoder)

    def forward(self, pt, t, set_emb, attn_mask=None):
        return self.decoder(pt, t, set_emb)

    def hungarian(self, predicted, target):
        N, L = predicted.shape
        distance_dict = {}

        for i in range(N):
            distance_dict[i] = {}
            for j in range(N):
                distance = torch.norm(predicted[i] - target[j], p=2).item()
                distance_dict[i][j] = distance

        total_distance = algorithm.find_matching(distance_dict, matching_type='min', return_type='total')

        return total_distance
    
    def get_loss(self, x_1, set_emb, attn_mask=None):
        x_0 = torch.randn_like(x_1).to(x_1.device)
        t = torch.rand(x_1.shape[0]).to(x_1.device) 
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        pred_vf = self(x_0, t, set_emb)
        loss = torch.pow(pred_vf - path_sample.dx_t, 2).mean()

        return loss
    
    def sample(self, batch_size, set_emb, timesteps, step_size, device):
        x_init = torch.randn((1, batch_size, 2), dtype=torch.float32, device=device)
        T = torch.linspace(0,1,timesteps).to(device)
        sol = self.solver.sample(
            time_grid=T, 
            x_init=x_init,
            method='midpoint', 
            step_size=step_size, 
            return_intermediates=True,
            enable_grad=False,
            model_extras=set_emb
        )

        return sol
