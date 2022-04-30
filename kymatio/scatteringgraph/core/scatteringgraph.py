# Authors:
# Scientific Ancestry:

def scatteringgraph(x, J, Q, psi, normalize, max_order, backend):
    absolute_value = backend.absolute_value
    concatenate = backend.concatenate
    matmul = backend.matmul
    moment = backend.moment
    sqrt = backend.sqrt

    # sizes of nth order coefficents of scattering transform
    order0_size = Q
    order1_size = Q * J
    order2_size = Q * (J * (J-1)) // 2


    # initalize arrays that will hold scattering coefficents
    out_S_0, out_S_1, out_S_2 = [], [], []
   
    # compute q moments
    # these are our zero order scattering coefficents
    for q in range(1, Q + 1):
        if normalize:   # normalized
            if q == 1:      # mean
                mu = moment(x, q)
                out_S_0.append(mu)
            elif q == 2:    # variance
                var = moment(x, q, mu)
                out_S_0.append(var)
                std = sqrt(var)
            else:           # skew, kurtosis, so on
                S_0_q = moment(x, q, mu, std)
                out_S_0.append(S_0_q)
        else:           # unnormalized
            S_0_q = moment(x, q)
        
            # add to array
            out_S_0.append(S_0_q)

    if max_order < 1:
        return concatenate(out_S_0)

    # compute first order coefficents
    for j_1 in range(0, J):
        # multiplication with graph wavelet filter
        U_1_c = matmul(psi[j_1], x)        
        
        # take absolute value for momement calculation
        U_1_a = absolute_value(U_1_c)

        for q in range(1, Q + 1):
            if normalize:   
                if q == 1:      
                    mu = moment(U_1_a, q)
                    out_S_1.append(mu)
                elif q == 2:    
                    var = moment(U_1_a, q, mu)
                    out_S_1.append(var)
                    std = sqrt(var)
                else:           
                    S_1_q = moment(U_1_a, q, mu, std)
                    out_S_1.append(S_1_q)
            else:           
                S_1_q = moment(U_1_a, q)
            
                out_S_1.append(S_1_q)

        if max_order < 2:
            continue

        # compute second order coefficents 
        for j_2 in range(j_1 + 1, J):
            # multiplication with different graph wavelet filter
            U_2_c = matmul(psi[j_2], U_1_a)
            
            U_2_a = absolute_value(U_2_c)

            for q in range(1, Q + 1):
                if normalize:   
                    if q == 1:      
                        mu = moment(U_1_a, q)
                        out_S_1.append(mu)
                    elif q == 2:    
                        var = moment(U_1_a, q, mu)
                        out_S_1.append(var)
                        std = sqrt(var)
                    else:           
                        S_1_q = moment(U_1_a, q, mu, std)
                        out_S_1.append(S_1_q)
                else:           
                    S_2_q = moment(U_2_a, q)
               
                    out_S_2.append(S_2_q)

    # collect all the scattering coeffcients into one tensor
    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)
    out_S = concatenate(out_S)
    return out_S


__all__ = ['scatteringgraph']
