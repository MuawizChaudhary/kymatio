import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def normalized_moment(x, q, mean=0, variance=1):
    "Calculate normalized moment"
    if isinstance(mean, int):
        mean = torch.zeros(1, x.shape[1]).to(x.device)

    if isinstance(variance, int):
        variance = torch.ones(1, x.shape[1]).to(x.device)

    x_mu = x - mean
    numerator = torch.pow(x_mu, q)
    denominator = torch.pow(variance, q)    
    ratio = torch.div(numerator, denominator)
    q_th_moment = torch.sum(ratio, dim=0) / x.shape[0]

    return q_th_moment.reshape(-1, 1)

def unnormalized_moment(x, q):
    "Calculate unnormalized moment"
    x_q = torch.pow(x, q)
    q_th_moment = torch.sum(x_q, dim=0)
    return q_th_moment.reshape(-1, 1)


backend = namedtuple('backend', ['name', 'normalized_moment', 'unnormalized_moment'])
backend.name = 'torch'
backend.normalized_moment = normalized_moment
backend.unnormalized_moment = unnormalized_moment
