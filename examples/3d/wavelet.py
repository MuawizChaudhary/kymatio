from kymatio.scattering3d.frontend.torch_frontend import WaveletTorch3D
import numpy as np
import torch
import glob
import pandas

orientations = np.array([(1,0,0), (0, 1,0), (0, 0, 1), (1,1,0), (0, 1,1), (1,
    0, 1), (1, 1, 1)])

# glob to get all the cubes
filenames = glob.glob('../../../*/*/df_m_64_z=0.npy')
filenames.sort()

data = []

# load cubes into numpy
for filename in filenames:
    data.append(np.load(filename))

data = np.array(data)

with torch.no_grad():
    S = WaveletTorch3D(3, (64, 64, 64), orientations, rho='relu', return_list=False,
            subsample=False).cuda().double()

    # number of training iterations x batch size x spatial dim
    data = data.reshape(-1, 2, 64, 64, 64)

    # total number of cubes x scattering features
    phase_harmonics = torch.empty((1000, 84)).cuda()

    counter = 0
    for i in data:
        # x.shape = batchsize x spatial dim
        x = torch.from_numpy(i).double().cuda()

        # Sx.shape =  batchsize x wavelet coefficents x phases x spatial dim
        Sx = S(x)

        backend = S.backend

        # mean.shape = batchsize x wavelet coefficents*phases
        mean = Sx.mean((-3, -2, -1)).reshape(Sx.shape[0], -1)
        phase_harmonics[counter : counter + i.shape[0]] = mean

        counter += i.shape[0]
    
    # centered_phase_harmonics.shape = dataset size x wavelet coefficents*phases
    centered_phase_harmonics = phase_harmonics - phase_harmonics.mean(0)

    # wavelet coefficents*phases x wavelet coefficents*phases
    cov = centered_phase_harmonics.t() @ centered_phase_harmonics

    # 500 x num_hyperparameters x calculation orientations x wavelet coefficents*phases
    phase_harmonics = phase_harmonics.reshape(500, 1, 2, -1)

    grad = phase_harmonics[..., 0, :] - phase_harmonics[..., 1, :]
    grad = grad.reshape(500, 84)

    cov_i = torch.inverse(cov + torch.eye(84).cuda())

    # 500 x 500
    # I should do this batch by batch perhaps
    fisher_1 = torch.matmul(cov_i, grad.t())
    fisher_2 = torch.matmul(grad, fisher_1)

