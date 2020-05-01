'''Defines non-normal initializers for vanilla RNNs in torch'''

import torch

def chain_init(n, diag_gain=0, offdiag_gain=1.02):
    return diag_gain * torch.eye(n) + offdiag_gain * torch.diag(torch.ones(n-1), diagonal=1)

def fbchain_init(n, diag_gain=0, offdiag_gain=0.04):
    return diag_gain * torch.eye(n) + 0.99 * torch.diag(torch.ones(n-1), diagonal=1) + \
           offdiag_gain * torch.diag(torch.ones(n-1), diagonal=-1)

def nonnormal_source_init(n, m, scale=0.9):
    return scale * torch.eye(n, m)
