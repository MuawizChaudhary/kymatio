
def wavelet_standard(x, pad, unpad, backend, J, L, phi, psi, max_order, rho,
        return_list, subsample):
    rfft = backend.rfft
    ifft = backend.ifft
    cdgmm3d = backend.cdgmm3d
    subsample_fourier = backend.subsample_fourier
    concatenate_3d  = backend.concatenate_3d

    order1_size = L * J
    output_size = order1_size

    out_U_1 = []

    U_r = pad(x)

    U_0_c = rfft(U_r)

    for n1 in range(len(psi)):
        j1 = psi[n1]['j']
        U_1_c = cdgmm3d(U_0_c, psi[n1][0])
        if j1 > 0 and subsample:
            U_1_c = subsample_fourier(U_1_c, k=2 ** j1)
        U_1_c = ifft(U_1_c)

        U_1_r = rho(U_1_c)
        print(U_1_r.shape)
        out_U_1.append(U_1_r)

    out_U = []
    out_U.extend(out_U_1)

    if not return_list and not subsample:
        out_U = concatenate_3d(out_U)

    return out_U
