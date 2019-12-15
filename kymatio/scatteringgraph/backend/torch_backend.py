import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def normalized_moment(x, q, mean=0, variance=1):
    "Calculate normalized moment"
    if isinstance(mean, int):
        mean = torch.zeros(1, x.shape[1]).to(x.device)

    if isinstance(variance, int):
        variance = torch.ones(1, x.shape[1]).to(x.device)

    diff = x - mean
    z_score = torch.div(diff, variance)
    z_score_q  = torch.pow(z_score, q)
    q_th_moment = torch.mean(z_score_q, dim=0)

    return q_th_moment.reshape(-1, 1)

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
    return torch.stack(arrays, axis=-3)

backend = namedtuple('backend', ['name', 'normalized_moment', 'unnormalized_moment', 'absolute_value', 'concatenate'])
backend.name = 'torch'
backend.normalized_moment = normalized_moment
backend.unnormalized_moment = unnormalized_moment
backend.absolute_value = absolute_value
backend.concatenate = concatenate
