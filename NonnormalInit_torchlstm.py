'''Defines non-normal initializers for LSTMs in torch'''

import torch
import numpy as np
from scipy.linalg import toeplitz

def chain_init(n, diag_gain=0, offdiag_gain=1.02):
    W = diag_gain * torch.eye(n) + offdiag_gain * torch.diag(torch.ones(n-1), diagonal=1)
    W = torch.cat((W, W, W, W), 0)
    return W

def fbchain_init(n, diag_gain=0, offdiag_gain=0.04):
    W = diag_gain * torch.eye(n) + 0.99 * torch.diag(torch.ones(n-1), diagonal=1) + offdiag_gain * torch.diag(torch.ones(n-1), diagonal=-1)
    W = torch.cat((W, W, W, W), 0)
    return W

def ramp_init(n, diag_gain=1.02, end_gain=0.01):
    W = toeplitz(np.linspace(0, -end_gain, n), np.linspace(0, end_gain, n))
    W = torch.FloatTensor(W)
    V = diag_gain * torch.diag(torch.ones(n-1), diagonal=1)  # update gate
    W = torch.cat((W, W, V, W), 0)
    return W

def nonnormal_source_init(n, m, scale=0.9):
    W = scale * torch.eye(n, m)
    W = torch.cat((W, W, W, W), 0)
    return W
