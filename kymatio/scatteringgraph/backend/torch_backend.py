import torch
from collections import namedtuple

BACKEND_NAME = 'torch'

def moment(x, q, mean=0, variance=1):
    "Calculate moment"
    if mean == 0:
        mean = torch.zeros(1, x.shape[1]).to(x.device())

    if variance == 1:
        variance = torch.ones(1, x.shape[1]).to(x.device())

    x_mu = x - mean
    numerator = torch.pow(x_mu, q)
    denominator = math.pow(variance, q) 
    ratio = torch.divide(numerator, denominator)
    q_th_moment = torch.sum(ratio, dim=1) / x.shape[0]

    return q_th_moment

backend = namedtuple('backend', ['name', 'moment'])
backend.name = 'torch'
backend.momment = moment
