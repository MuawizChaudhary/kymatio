from kymatio.scattering3d.frontend.torch_frontend import WaveletTorch3D
import numpy as np
import torch

orientations = np.array([(1,0,0), (0, 1,0), (0, 0, 1), (1,1,0), (0, 1,1), (1,
    0, 1), (1, 1, 1)])

x = torch.zeros((1, 64, 64, 64)).cuda().double()
x[:, 16:48, 16:48, 16:48] = 1

S = WaveletTorch3D(3, (64, 64, 64), orientations, rho='relu', return_list=False,
        subsample=False).cuda().double()
Sx = S(x)
print(len(Sx))
print(Sx.shape)
