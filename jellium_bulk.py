"""Set of functions to obtain dielectric properties of bulk jellium"""

import math
import numpy as np
import tools
import numba
from numba import jit

ic = complex(0,1)
E0 = 1/(4*math.pi)


def im_chi0(q_vec, omega, dens=0.025):
    """Computes the imaginary part of the RPA density response function given by Lindhard"""
    npnt = len(q_vec) #size of the sampling (keep in mind that periodicity is
                  #induced in real space from the discrete sampling)
    n_w = len(omega) #Number of frequency sampled
    chi0_q = np.zeros((n_w, npnt), dtype = "c16")
    e_f = (1/2)*(3*math.pi**2*dens)**(2/3)
    for i in range(n_w):
        w_i = omega[i]
        for j in range(npnt):
            q_norm = abs(q_vec[j])
            if q_norm==0:
                continue
            e_minus = (w_i-(q_norm**2)/2)**2*(2/(q_norm**2))*1/4
            if e_minus<=(e_f-w_i):
                chi0_q[i, j]=1/(2*math.pi)*(1/q_norm)*w_i
            elif e_f >= e_minus >= e_f-w_i:
                chi0_q[i, j]=1/(2*math.pi)*(1/q_norm)*(e_f-e_minus)
            else:
                continue
    return -chi0_q


def re_chi0(q_vec, omega, dens=0.025):
    """Computes the imaginary part of the RPA density response function given by Lindhard"""
    npnt = len(q_vec)
    n_w = len(omega)
    chi0_q = np.zeros((n_w, npnt), dtype = "c16")
    k_f = (3*math.pi**2*dens)**(1/3)
    pref = -4/(2*math.pi)**2*k_f
    for i in range(n_w):
        w_i = omega[i]
        k_w = (2*w_i)**(1/2)
        for j in range(npnt):
            q_norm = abs(q_vec[j])
            if q_norm==0:
                continue
            t_1 = 1-(1/4)*((k_w**2-q_norm**2)**2/(k_f**2*q_norm**2))
            t_21 = (k_w**2-2*k_f*q_norm-q_norm**2)/(k_w**2+2*k_f*q_norm-q_norm**2)
            t_22 = math.log(abs(t_21))
            t_3 = 1-(1/4)*((k_w**2+q_norm**2)**2/(k_f**2*q_norm**2))
            t_41 = (k_w**2+2*k_f*q_norm+q_norm**2)/(k_w**2-2*k_f*q_norm+q_norm**2)
            t_42 = math.log(abs(t_41))
            chi0_q[i, j] = 1/2+(k_f/(4*q_norm))*(t_1*t_22+t_3*t_42)
    return pref*chi0_q


def chi0q(q_vec, omega, dens = 0.025):
    """Computes the RPA density response function in reciprocal space
     following the Lindhard formula"""
    rchi = re_chi0(q_vec, omega, dens)
    ichi = im_chi0(q_vec, omega, dens)
    chi0_q_in = rchi+ic*ichi
    chi0_q_out = chi0_mat(chi0_q_in)
    return chi0_q_out

@jit(nopython = True, parallel=True)
def chi0_mat(chi0_q):
    """Transform chi0_q (shape: n_w, nq) in a chi0qq
     (serie of diagonal matrices; shape: n_w, nq, nq)"""
    n_w, n_q = chi0_q.shape
    chi0_qq = np.zeros((n_w, n_q, n_q), dtype = "c16")
    for i in range(n_w):
        chi0_qq[i] = np.diag(chi0_q[i])
    return chi0_qq   



@jit(nopython = True, parallel=True)
def chiq_mat(chi0_q, q_vec):
    """Computes the density response function with coulomb effect with a matrix as input."""
    n_w, n_q = chi0_q.shape[0], chi0_q.shape[1]
    chiq_m = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    for i in range(n_q):
        if q_vec[i] == 0:
            continue
        coulomb[i, i] = 4*math.pi/(q_vec[i]**2)
    #print(np.diag(coulomb))
    print("coulomb ok")
    chi_to_inv = np.zeros((n_q, n_q), dtype = "c16")
    for i in range(n_w):
        for j in range(n_q):
            if chi0_q[i, j, j] == 0:
                chiq_m[i, j, j] = chi0_q[i, j, j]
            else:
                chi_to_inv[j, j] = (chi0_q[i, j, j])**(-1)-coulomb[j, j]
                chiq_m[i, j, j] = chi_to_inv[j, j]**(-1)
    return chiq_m

@jit(nopython = True, parallel=True)
def epsilon(chi0qgg, q_vec):
    """Computes the dielectric response of the given density response function"""
    n_w, n_q= chi0qgg.shape[0], chi0qgg.shape[1]
    eps_out = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    for i in range(n_q):
        if q_vec[i]==0:
            continue
        coulomb[i, i] = 4*math.pi/(q_vec[i]**2)
    for i in range(n_w):
        eps_out[i] = np.diag(np.ones(n_q))-np.multiply(coulomb, chi0qgg[i])
    return eps_out


def fourier_dir_wqq(matwqq, d):
    """Performs the Fourier Transform to go from chi0wqq to chi0wzz"""
    n_w, nq1, nq2 = matwqq.shape
    matwzq2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq1):
            matwzq2[i, j, :] = np.fft.fft(matwqq[i, j, :])
    matwz1z2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq2):
            matwz1z2[i, :, j] = np.fft.fft(matwzq2[i, :, j])
    matwz1z2 = matwz1z2
    matwz1z2_out = np.zeros((n_w, nq1, nq2), dtype = "c16")
    """for i in range(n_w):
        matwz1z2_out[i] = (matwz1z2[i, :, :]+np.transpose(matwz1z2[i, :, :]))/2"""
    return matwz1z2/d 


#A Refaire sans doute
"""@jit(nopython = True, parallel=True)
def chiq(chi0_q, q_vec, q_p, opt = "positive"):
    Computes the density response function with coulomb effect with a vector as input.
    Option are "positive" with q values from 0 to q_max or "symmetric" with values
    from -q_max to q_max
    n_w, n_q = chi0_q.shape[0], chi0_q.shape[1]
    chiq_m = np.zeros((n_w, n_q), dtype = "c16")
    coulomb = np.zeros((n_q))
    if opt == "positive" and q_p != 0:
        coulomb = 4*math.pi*np.power(np.abs(q_vec+q_p), -2)
    elif opt == "positive" and q_p == 0:
        coulomb[1:n_q] = 4*math.pi*np.power(np.abs(q_vec[1:n_q]), -2)
    elif opt == "symmetric" and q_p == 0:
        q_vec = np.real(tools.inv_rev_vec(q_vec))
        coulomb[1:n_q] = 4*math.pi*np.power(np.abs(q_vec[1:n_q]), -2)
    else:
        q_vec = np.real(tools.inv_rev_vec(q_vec))
        coulomb = 4*math.pi*np.power(np.abs(q_vec+q_p), -2)
    for i in range(n_w):
        for j in range(1, n_q):
            chiq_inv = np.power(np.diag(chi0_q[i])[j], -1, dtype="c16")-coulomb[j]
            chiq_m[i, j] = np.power(chiq_inv, -1, dtype="c16")
    return chiq_m"""