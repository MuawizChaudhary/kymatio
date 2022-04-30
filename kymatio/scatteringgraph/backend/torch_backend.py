import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def matmul(A, B):
    "Matrix Multiplication"
    return torch.matmul(A, B)

def normalized_moment(x, q, mean=0, std=1):
    "Calculate normalized moment"
    if isinstance(mean, int):
        mean = torch.mean(x, 0).reshape(-1, 1)
        return mean
    diff = x - mean
    if isinstance(std, int):
        std = torch.std(diff, 0, False).reshape(-1, 1)
        return std
    z_score = torch.div(diff, std)
    z_score[z_score != z_score]  = 0
    z_score_q  = torch.pow(z_score, q)
    q_th_moment = torch.mean(z_score_q, dim=0)
    if q == 4:
        q_th_moment = q_th_moment - 3
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
