import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def moment(x, q, mean=0, variance=1):
    "Calculate moment"
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

backend = namedtuple('backend', ['name', 'moment'])
backend.name = 'torch'
backend.moment = moment
