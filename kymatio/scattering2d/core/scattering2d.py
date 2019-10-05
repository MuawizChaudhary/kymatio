# Authors: Edouard Oyallon
# Scientific Ancestry: Edouard Oyallon, Laurent Sifre, Joan Bruna


__all__ = ['scattering2d']

def scattering2d(x, pad, unpad, backend, J, L, phi, psi, max_order):
    subsample_fourier, modulus, fft, cdgmm, finalize = backend.subsample_fourier, backend.modulus,\
                                                         backend.fft, backend.cdgmm, backend.finalize
    order0_size = 1
    order1_size = L * J
    order2_size = L ** 2 * J * (J - 1) // 2
    output_size = order0_size + order1_size

    if max_order == 2:
        output_size += order2_size

    out_S_0, out_S_1, out_S_2 = [], [], []
    U_r = pad(x)
    U_0_c = fft(U_r, 'C2C')

    # First low pass filter
    U_1_c = subsample_fourier(cdgmm(U_0_c, phi[0]), k=2**J)

    S_0 = fft(U_1_c, 'C2R', inverse=True)

    out_S_0.append(unpad(S_0))

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm(U_0_c, psi[n1][0])
        if(j1 > 0):
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = fft(U_1_c, 'C2C', inverse=True)
        U_1_c = fft(modulus(U_1_c), 'C2C')

        # Second low pass filter
        S_1_c = subsample_fourier(cdgmm(U_1_c, phi[j1]), k=2**(J-j1))
        S_1_r = fft(S_1_c, 'C2R', inverse=True)
        out_S_1.append(unpad(S_1_r))

        if max_order == 2:
            for n2 in range(len(psi)):
                j2 = psi[n2]['j']
                if(j1 < j2):
                    U_2_c = subsample_fourier(cdgmm(U_1_c, psi[n2][j1]), k=2 ** (j2-j1))
                    U_2_c = fft(U_2_c, 'C2C', inverse=True)
                    U_2_c = fft(modulus(U_2_c), 'C2C')
                    # Third low pass filter
                    S_2_c = subsample_fourier(cdgmm(U_2_c, phi[j2]), k=2 ** (J-j2))
                    S_2_r = fft(S_2_c, 'C2R', inverse=True)

                    out_S_2.append(unpad(S_2_r))

    out_S = finalize(out_S_0, out_S_1, out_S_2)
    return out_S
