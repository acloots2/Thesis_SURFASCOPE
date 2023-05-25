"""Solving Schrodinger"""

import numpy as np
import tools


def ecin(g_vec):
    g_vec = np.real(tools.inv_rev_vec(g_vec))
    #print(g_vec)
    return np.diag(g_vec**2/2)


def epot(v_x):
    v_g = np.fft.ifft(v_x)
    n_g = len(v_g)
    v_gmat = np.zeros((n_g, n_g), dtype = "c16")
    for i in range(n_g):
        for j in range(n_g):
            v_gmat[i, j] = v_g[j-i]   
    return v_gmat


def Hamitonian(v_x, z_vec):
    g_v = tools.zvec_to_qvec(z_vec)
    e_cin = ecin(g_v)
    e_pot = epot(v_x)
    return e_cin+e_pot


def eig_energie(v_x, z_vec):
    ham = Hamitonian(v_x, z_vec)
    eig_v, eig_f = np.linalg.eig(ham)
    return eig_v, eig_f 

