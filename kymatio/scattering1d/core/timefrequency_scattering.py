import math 
def timefrequency_scattering(x, pad, unpad, backend, J, J_fr, psi1, psi2, phi, 
                             psi_fr, phi_fr, T="global", pad_left=0,pad_right=0, ind_start=None, 
                             ind_end=None, oversampling=0, size_scattering=(0, 0, 0), 
                             out_type='array'):

    subsample_fourier = backend.subsample_fourier
    transpose = backend.transpose
    modulus = backend.modulus
    mean = backend.mean
    rfft = backend.rfft
    ifft = backend.ifft
    irfft = backend.irfft
    cdgmm = backend.cdgmm
    concatenate = backend.concatenate
    to_real = backend.to_real
    real_out = backend.real_out 

    U_0 = pad(x, pad_left, pad_right)
    U_0_hat = rfft(U_0)

    # good spot to print shapes
    U_1_list = []
    S_1_T_list = []
    for n1 in range(len(psi1)):
        j1 = psi1[n1]['j']
        k1 = max(j1 - oversampling, 0)

        U_1_c = cdgmm(U_0_hat, psi1[n1][0])
        U_1_hat = subsample_fourier(U_1_c, 2**k1)
        U_1_c = ifft(U_1_hat)

        U_1_m = modulus(U_1_c)
        U_1_hat = rfft(U_1_m)

        U_1_list.append(U_1_hat)

        k1_J = max(J - k1 - oversampling, 0)

        if T == "global":
            S_1_T = mean(U_1_m)
            S_1_T = real_out(S_1_T)
        else:
            S_1_c = cdgmm(U_1_hat, phi[k1])
            S_1_hat = subsample_fourier(S_1_c, 2**k1_J)
            S_1_r = irfft(S_1_hat)

            S_1_T = unpad(S_1_r, ind_start[k1_J + k1], ind_end[k1_J + k1])
            # good spot to print shapes
        S_1_T_list.append(S_1_T)

    total_height = 2 ** math.ceil(1+math.log2(len(psi1)))
    padding_row = 0 * S_1_T
    for n1 in range(total_height - len(S_1_T_list)):
        S_1_T_list.append(padding_row)
    S_1_TM = to_real(concatenate(S_1_T_list))

    k_fr_J = max(J_fr - oversampling, 0)
    S_1_TM_T = transpose(S_1_TM)
    S_1_TM_T_hat = rfft(S_1_TM_T)
    S_1_TM_T_c = cdgmm(S_1_TM_T_hat, phi_fr[0])
    S_1_TM_T_hat = subsample_fourier(S_1_TM_T_c, 2**k_fr_J)
    S_1_TM_T = irfft(S_1_TM_T_hat)
    S_1_FR = transpose(S_1_TM_T)
    S_1_FR = real_out(S_1_FR)
    # good spot to print shapes

    S_2_list = []
    for n2 in range(len(psi2)):
        j2 = psi2[n2]['j']
        if j2 == 0:
            continue
        U_2_list = []

        for n1 in range(len(psi1)):
            j1 = psi1[n1]['j']
            if j1 >= j2:
                continue
            k1 = max(j1 - oversampling, 0)
            k2 = max(j2 - j1 - oversampling, 0)

            U_1_hat = U_1_list[n1]

            U_2_c = cdgmm(U_1_hat, psi2[n2][k1])
            U_2_hat = subsample_fourier(U_2_c, 2**k2)
            U_2_c = ifft(U_2_hat)
            U_2_list.append(U_2_c)
            # good spot to print shapes

        padding_row = 0 * U_2_c 
        for n in range(total_height - len(U_2_list)):
            U_2_list.append(padding_row)
        U_2 = concatenate(U_2_list)

        U_2_T = transpose(U_2)
        U_2_hat_T = ifft(U_2_T)
        # good spot to print shapes
        
        for n_fr in range(len(psi_fr)):
            j_fr = psi_fr[n_fr]['j']
            k_fr = max(j_fr - oversampling, 0)
            U_fr_c = cdgmm(U_2_hat_T, psi_fr[n_fr][0])
            U_fr_hat = subsample_fourier(U_fr_c, 2**k_fr)

            U_2_m = modulus(U_fr_hat)
            
            k_J_fr = max(J_fr  - k_fr - oversampling, 0)
            U_2_hat = rfft(U_2_m)
            S_2_fr_c = cdgmm(U_2_hat, phi_fr[k_fr])
            S_2_fr_hat = subsample_fourier(S_2_fr_c, 2**k_J_fr)
            S_2_fr = irfft(S_2_fr_hat)
            S_2_fr = transpose(S_2_fr)

            if T == "global":
                S_2 = mean(S_2_fr)
                S_2 = real_out(S_2)
            else:
                k2_J = max(J - j2 - oversampling, 0)
                U_2_hat = rfft(S_2_fr)
                S_2_c = cdgmm(U_2_hat, phi[j2])
                S_2_hat = subsample_fourier(S_2_c, 2 ** k2_J)
                S_2_r = irfft(S_2_hat)
                S_2 = unpad(S_2_r, ind_start[k2_J+k2+k1], ind_end[k2_J+k2+k1])
            # good spot to print shapes
            S_2_list.append(S_2)

    out_S = []
    #out_S.extend([S_1_FR])
    out_S.extend(S_2_list)
    out_S = concatenate(out_S)
    # good spot to print shapes
    return out_S


