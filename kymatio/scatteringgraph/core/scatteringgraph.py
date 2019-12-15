# Authors:
# Scientific Ancestry:

def scatteringgraph(x, J, Q, A, psi, normalize, max_order, backend):
    absolute_value = backend.absolute_value
    moment = backend.moment

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
            return
        else:           # unnormalized
            S_0_q = moment(x, q)
        
        # add to array
        out_S_0.append(S_0_q)

    if max_order >= 1:
        continue
    # compute first order coefficents
    for j_1 in range(0, J):
        # multiplication with graph wavelet filter
        U_1_c = psi[j_1] * x        

        # take resulting absolute value, we want this for unnormalized momement
        # calculation
        if max_order >= 2 or not normalize:
            U_1_a = absolute_value(U_1_c)

        # compute q moments
        for q in range(1, Q + 1):
            if normalize:   
                return       
            else:           
                S_1_q = moment(U_1_a, q)
            
            out_S_1.append(S_1_q)


        if max_order < 2:
            continue

        # compute second order coefficents 
        for j_2 in range(j_1 + 1, J):
            # multiplication with different graph wavelet filter
            U_2_c = psi[j_2] * U_1_a
            
            if not normalize:
                U_2_a = absolute_value(U_2_c)

            # compute q moments
            for q in range(1, Q + 1):
                if normalize:   
                    return      
                else:           
                    S_2_q = moment(U_2_a, q)
               
                out_S_2.append(S_2_q)
    
    out_S = []
    out_S.extend(out_S_0)
    out_S.extend(out_S_1)
    out_S.extend(out_S_2)

    out_S = concatenate(out_S)
    return out_S


__all__ = ['scatteringgraph']
            
