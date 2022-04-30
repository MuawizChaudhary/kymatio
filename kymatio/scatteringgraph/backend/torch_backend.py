import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def matmul(A, B):
    "Matrix Multiplication"
    return torch.matmul(A, B)

def normalized_moment(x, q, mean=0, std=1):
    "Calculate normalized moment"
    if isinstance(mean, int):
        mean = torch.zeros(1, x.shape[1]).to(x.device)

    if isinstance(std, int):
        std = torch.ones(1, x.shape[1]).to(x.device)

    diff = x - mean
    z_score = torch.div(diff, std)
    z_score_q  = torch.pow(z_score, q)
    q_th_moment = torch.mean(z_score_q, dim=0)
    return q_th_moment.reshape(-1, 1)

def sqrt(x):
    "Calculated square root"
    return torch.sqrt(x)

def unnormalized_moment(x, q):
    "Calculate unnormalized moment"
    x_q = torch.pow(x, q)
    q_th_moment = torch.sum(x_q, dim=0)
    return q_th_moment.reshape(-1, 1)

def absolute_value(x):
    "Calculate absolute value"
    return torch.abs(x)

def concatenate(arrays):
    "Concatenate arrays together at end"
    return torch.stack(arrays, axis=0)

backend = namedtuple('backend', ['name', 'matmul', 'normalized_moment',
    'unnormalized_moment', 'absolute_value', 'concatenate', 'sqrt'])
backend.name = 'torch'
backend.matmul = matmul
backend.normalized_moment = normalized_moment
backend.unnormalized_moment = unnormalized_moment
backend.absolute_value = absolute_value
backend.concatenate = concatenate
backend.sqrt = sqrt
