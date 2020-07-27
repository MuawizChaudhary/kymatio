from kymatio.scattering3d.frontend.torch_frontend import WaveletTorch3D
import numpy as np
import torch

orientations = np.array([(1,0,0), (0, 1,0), (0, 0, 1), (1,1,0), (0, 1,1), (1,
    0, 1), (1, 1, 1)])

x = torch.randn((1, 64, 64, 64)).cuda().double() #torch.zeros((1, 64, 64, 64)).cuda().double()
#x[:, 16:48, 16:48, 16:48] = 1

y = torch.autograd.Variable(x, requires_grad=True)

S = WaveletTorch3D(3, (64, 64, 64), orientations, rho='relu', return_list=False,
        subsample=False).cuda().double()

Sx = S(y)
backend = S.backend

mean = backend.mean(Sx).reshape(Sx.shape[0], -1)

cov = backend.covariance(mean, torch.mean(mean, dim=-1, keepdim=True))
i_cov = torch.inverse(cov)

#fisher_1 = torch.matmul(i_cov, grad)
#fisher_2 = torch.matmul(grad.t(), fisher_1)
