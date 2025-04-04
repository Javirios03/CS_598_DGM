import time
import torch

from torch import nn, Tensor

# flow_matching
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.path import AffineProbPath
from flow_matching.solver import Solver, ODESolver
from flow_matching.utils import ModelWrapper

from torchdiffeq import odeint
# from chamferdist import ChamferDistance

# visualization
# import matplotlib.pyplot as plt

from matplotlib import cm
import warnings

from hungarian_algorithm import algorithm

warnings.filterwarnings("ignore", category=UserWarning, module='torch')

class Flow(nn.Module):
    def __init__(self, config, decoder):
        super().__init__()
        self.decoder = decoder
        self.path = AffineProbPath(scheduler=CondOTScheduler())
        self.solver = ODESolver(velocity_model=self.decoder)
        self.eps = 1e-5

    def velocity(self, x, t):
        mu = self.decoder(x, t, model_extras=None)
        return (mu - x) / (1. - t.unsqueeze(-1) + self.eps)

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
        # x_1 = x_1.squeeze(0)
        x_0 = torch.randn_like(x_1).to(x_1.device)
        # print("x_1: ", x_1.shape)
        t = torch.rand(x_1.shape[0]).to(x_1.device) 
        # print("t: ", t.shape)
        path_sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        # print("path sample", path_sample.x_t.shape)
        pred_mu = self(path_sample.x_t, path_sample.t, set_emb)
        # print("pred_vf" , pred_vf.shape)
        loss = 0.5 * torch.pow(pred_mu - x_1, 2).sum()

        return loss

    def sample(self, batch_size, device, steps=50, set_emb=None):
        x_0 = torch.randn(1, batch_size, 2).to(device)
        device = x_0.device
        time_grid = torch.linspace(0., 1., steps, device=device)

        def ode_func(t, x):
            t_batch = t * torch.ones(x.shape[0], device=device)
            v = self.velocity(x, t_batch)
            return v

        sol = odeint(
            ode_func,
            x_0,
            time_grid,
            method="euler",
        )

        return sol[-1]