import numpy as np
#from numpy import linalg as LA
import numpy.linalg as LA
import scipy.stats.mstats

def lazy_random_walk(A):
    P_array = []
    d = A.sum(0)
    P_t = A/d
    P_t[np.isnan(P_t)] = 0
    P = 1/2*(np.identity(P_t.shape[0])+P_t)
    return P

def graph_wavelet(P):
    psi = []
    for d1 in [1,2,4,8,16]:
        W_d1 = LA.matrix_power(P,d1) - LA.matrix_power(P,2*d1)
        psi.append(W_d1.astype(np.float32))
    return psi

def normalized_zero_order_feature(ro):
    F0 = []
    mu = np.mean(ro,0)
    F0.append(mu)
    var = np.var(ro,0)
    F0.append(var)
    skew = scipy.stats.skew(ro,bias=0,axis=0)
    F0.append(skew)
    kurtosis = scipy.stats.kurtosis(ro,axis=0)
    F0.append(kurtosis)
    F0 = np.array(F0).reshape(-1,1)
    return F0

def normalized_first_order_feature(u):
    F1  = []
    W_u = u
    for u in W_u:
        mu = np.mean(u,0)
        F1.append(mu)
        var = np.var(u,0)
        F1.append(var)
        skew = scipy.stats.skew(u,bias=False,axis=0)
        F1.append(skew)
        kurtosis = scipy.stats.kurtosis(u,axis=0)
        F1.append(kurtosis)
    F1 = np.array(F1).reshape(-1, 1)
    return F1

def normalized_selected_second_order_feature(W,u):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = []
    W_u = u1
    for u1 in W_u:
        mu = np.mean(u1,0)
        F2.append(mu)
        var = np.var(u1,0)
        F2.append(var)
        skew = scipy.stats.skew(u1,bias=True,axis=0)
        F2.append(skew)
        kurtosis = scipy.stats.kurtosis(u1,axis=0)
        F2.append(kurtosis)
    F2 = np.array(F2).reshape(-1,1)
    return F2

def generate_graph_feature(A,ro):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    P = lazy_random_walk(A)
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = normalized_zero_order_feature(ro,f)
    F1 = normalized_first_order_feature(u,f)
    F2 = normalized_selected_second_order_feature(W,u,f)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    return F

def zero_order_feature(ro):
    F0 = []
    for i in [1,2,3,4]:
        F0.append(np.sum(np.power(np.abs(ro),i),0))
    F0 = np.array(F0).reshape(-1,1)
    return F0

def first_order_feature(u):
    F1  = []
    for i in [1,2,3,4]:
        F1.append(np.sum(np.power(u,i),1))
    F1 = np.array(F1).reshape(-1,1)
    return F1

def selected_second_order_feature(W,u):
    u1 = np.einsum('ij,ajt ->ait',W[1],u[0:1])
    for i in range(2,len(W)):
        u1 = np.concatenate((u1,np.einsum('ij,ajt ->ait',W[i],u[0:i])),0)
    u1 = np.abs(u1)
    F2 = []
    for i in [1,2,3,4]:
        F2.append(np.sum(np.power(u1,i),1))
    F2 = np.array(F2).reshape(-1,1)
    return F2

def generate_mol_feature(A,ro):
    #with zero order, first order and second order features
    #shall consider only zero and first order features
    P = lazy_random_walk(A)
    W = graph_wavelet(P)
    u = np.abs(np.matmul(W,ro))
    
    F0 = zero_order_feature(ro)
    F1 = first_order_feature(u)
    F2 = selected_second_order_feature(W,u)
    F = np.concatenate((F0,F1),axis=0)
    F = np.concatenate((F,F2),axis=0)
    return F
