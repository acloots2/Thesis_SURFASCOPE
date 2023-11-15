"""Set of functions to obtain dielectric properties of bulk jellium"""

import math
import cmath
import numpy as np
import numba
from numba import jit
import Sol_Schrod as ss
import tools



ic = complex(0,1)
E0 = 1/(4*math.pi)


def epsilon(chi0qgg, q_vec, q_p):
    """Computes the dielectric function of a slab associated with chi0qgg"""
    n_w, n_q = chi0qgg.shape[0], chi0qgg.shape[1]
    eps_out = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    if q_p == 0:
        for i in range(1, n_q):
            if q_vec[i]==0:
                continue
            coulomb[i, i] = 4*math.pi*np.power((q_vec[i]), -2)
    else:
        for i in range(n_q):
            coulomb[i, i] = 4*math.pi/(q_vec[i]**2+q_p**2)
    for i in range(n_w):
        eps_out[i] = np.diag(np.ones(n_q))-np.matmul(coulomb, chi0qgg[i])
    return eps_out


def fourier_inv(chi0wzz, z_vec):
    """Computes chi0qgg from chi0wzz"""
    n_w, nz1, nz2 = chi0wzz.shape
    if nz1 !=nz2:
        raise ValueError("The matrix must have the same dimension nz1 and nz2")
    chi0wzq2 = np.zeros((n_w, nz1, nz2), dtype = "c16")
    for i in range(n_w):
        for j in range(nz1):
            chi0wzq2[i, j, :] = np.fft.fft(chi0wzz[i, j, :], norm = "ortho")
    chi0wq1q2 = np.zeros((n_w, nz1, nz2), dtype = "c16")
    for i in range(n_w):
        for j in range(nz2):
            chi0wq1q2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j], norm = "ortho")
    return chi0wq1q2/nz2*max(z_vec)**2

@jit(nopython = True, parallel=True)
def sym_chi_slab(chi0wzz):
    """Symmetrizes chi0wzz of the first spatial components only spans in 
    the first half of the sampled area"""
    n_w, n_z1, n_z2 = chi0wzz.shape
    if n_z1 == n_z2:
        return chi0wzz
    else:
        chi0wzz_slab = np.zeros((n_w, n_z2, n_z2), dtype = "c16")
        for i in range(n_w):
            chi0wzz_slab[i, 0:n_z1, 0:n_z2] = chi0wzz[i, :, :]
            for j in range(n_z1):
                chi0wzz_slab[i, n_z1-1+j, :] = chi0wzz[i, n_z1-1-j, :][::-1]
        return chi0wzz_slab


 


@jit(nopython = True, parallel=True)
def sym_chi_slab_with_void(chi0wzz, z_2, void = 30):
    n_w, nz1, nz2 = chi0wzz.shape
    step = max(z_2)/nz2
    nstep_void = math.ceil(void/step)
    if nz1 == nz2:
        return chi0wzz
    else:
        chi0wzz_slab_wo_void = np.zeros((n_w, nz2, nz2), dtype = "c16")
        for i in range(n_w):
            chi0wzz_slab_wo_void[i, 0:nz1, 0:nz2] = chi0wzz[i, :, :]
            for j in range(nz1):
                chi0wzz_slab_wo_void[i, nz1-1+j, :] = chi0wzz[i, nz1-1-j, :][::-1]
        
        if nstep_void%2 == 0:
            chi0wzz_slab = np.zeros((n_w, nz2+nstep_void, nz2+nstep_void), dtype = "c16")
            nstep_half = math.floor(nstep_void/2)
            for i in range(n_w):
                chi0wzz_slab[i, nstep_half:nz2+nstep_void-nstep_half, nstep_half:nz2+nstep_void-nstep_half] = chi0wzz_slab_wo_void[i]
        else:
            nstep_void = nstep_void+1
            nstep_half = math.floor(nstep_void/2)
            chi0wzz_slab = np.zeros((n_w, nz2+nstep_void, nz2+nstep_void), dtype = "c16")
            for i in range(n_w):
                chi0wzz_slab[i, nstep_half:nz2+nstep_void-nstep_half, nstep_half:nz2+nstep_void-nstep_half] = chi0wzz_slab_wo_void[i]
        return chi0wzz_slab


@jit(nopython = True, parallel=True)
def chi0wzz_slab_jellium(q_p, z_1, z_2, omega, dens, d_slab, eta, nband):
    """Computes the density response function as found by Eguiluz with the slab represented as an infinite well"""
    print("Computation with IBM model started")
    n_w = int
    n_z1, n_z2 = int, int
    n_w = len(omega)
    n_z1, n_z2 = len(z_1), len(z_2)
    e_f = "c16"
    nmax = int
    chi0wzz = np.zeros((n_w, n_z1, n_z2), dtype = "c16")
    energies = np.zeros((nband))
    for i in range(1, nband):
        energies[i] = 1/2*(i**2*math.pi**2)/d_slab**2
    e_f, nmax = ef_2D_full(dens, d_slab, energies)
    wf1 = np.zeros((n_z1, nband), dtype = float)
    alpha_band = np.zeros(nband)
    for j in range(1, nband):
        alpha_band[j] = math.pi*j/d_slab
        wf1[:, j] = np.sin(alpha_band[j]*z_1)
    wf2 = np.zeros((n_z2, nband), dtype = float)
    for j in range(1, nband):
        wf2[:, j] = np.sin(alpha_band[j]*z_2)
    wff = np.zeros((n_z1, n_z2, nband), dtype = float)
    for i in range(n_z1):
        for j in range(n_z2):
            for k in range(1, nband):
                wff[i, j, k] = wf1[i, k]*wf2[j,k]
    wff = wff/np.linalg.norm(wf2[:, 1])**2
    wffi = float
    wffj = float
    for i in range(n_w):
        fll = np.zeros((nmax, nband), dtype = "c16")
        for j in range(1, nmax):
            for k in range(1, nband):
                fll[j, k] = f_ll(q_p, omega[i], j, k, d_slab, e_f, eta)
        for j in range(n_z1):
            for k in range(n_z2):
                for l in range(1, nmax):
                    wffi = wff[j, k, l]
                    for m in range(1, nband):
                        wffj = wff[j, k, m]
                        chi0wzz[i, j, k]+=wffi*wffj*fll[l, m] 
    return chi0wzz*n_z2**2/d_slab**2




@jit(nopython = True)
def e_l(l_i, d_slab):
    """Energy level in an infinite well"""
    return 1/2*l_i**2*math.pi**2/d_slab**2
@jit(nopython = True)
def a_ll(q_p, l_1, l_2, d_slab):
    """Segment of the prefactor of Eguiluz1985"""
    return (q_p**2)/2-(e_l(l_1, d_slab)-e_l(l_2, d_slab))


@jit(nopython = True)
def f_ll(q_p, omega, l_1, l_2, d_slab, e_f, eta) -> float:
    """Compute the prefactor from the formula of Eguiluz1985"""
    a_l1l2 = a_ll(q_p, l_1, l_2, d_slab)
    if q_p == 0:
        pre_factor = (e_f-e_l(l_1, d_slab))/(math.pi)
        return -pre_factor*(1/(a_l1l2+omega+ic*eta)+1/(a_l1l2-omega-ic*eta))
    else:
        k_l = cmath.sqrt(2*(e_f-e_l(l_1, d_slab)))
        return -1/(math.pi*q_p**2)*(2*a_l1l2+ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2-omega-ic*eta)**2)-ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2+omega+ic*eta)**2))
    
@jit(nopython = True)
def ef_2D(n, d)-> tuple[float, int]:
    "Find the fermi level in a slab"
    n = n*d
    if n == 0:
        raise ValueError("the fermi level is not uniquely defined if the density is zero")
    e_max = 0.1
    e_min = 0
    e_tot = 0
    i = 1
    while e_max > e_min:
        e_max = (math.pi*n + e_tot)/i
        i += 1
        e_min = i**2*math.pi**2/(2*d**2)
        e_tot += e_min
    return e_max, i

@jit(nopython = True)
def ef_2D_full(n, d, e_vec):
    "Find the fermi level in a slab"
    n = n*d
    if n == 0:
        raise ValueError("the fermi level is not uniquely defined if the density is zero")
    e_max = e_vec[0]+0.1
    e_min = e_vec[0]
    e_tot = e_vec[0]
    i = 1
    while np.real(e_max) > np.real(e_min):
        e_max = (math.pi*n + e_tot)/i
        if i > len(e_vec):
            raise ValueError("Number of states too low, you should add more states in order to find the Fermi level")
        e_min = e_vec[i]
        e_tot += e_min
        i += 1
    return e_max, i-1


def fourier_dir(epsqwgg):
    """Performs the Fourier Transform to go from chi0wqq to chi0wzz"""
    n_w, nq1, nq2 = epsqwgg.shape
    chi0wzq2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq1):
            chi0wzq2[i, j, :] = np.fft.ifft(epsqwgg[i, j, :])
    chi0wz1z2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq2):
            chi0wz1z2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j])
    chi0wz1z2 = chi0wz1z2*nq2
    chi0wz1z2_out = np.zeros((n_w, nq1, nq2), dtype = "c16")    
    for i in range(n_w):
        chi0wz1z2_out[i] = (chi0wz1z2[i, :, :]+np.transpose(chi0wz1z2[i, :, :]))/2
    return chi0wz1z2_out

@jit(nopython = True, parallel=True)
def chi_jellium_slab_test0(chi0qgg, q_vec, q_p):
    """Computes the interacting density response function"""
    n_w, n_q = chi0qgg.shape[0], chi0qgg.shape[1]
    chi_out = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    if q_p == 0:
        for i in range(1, n_q):
            if q_vec[i]==0:
                continue
            coulomb[i, i] = 4*math.pi*np.power((q_vec[i]), -2)
    else:
        for i in range(n_q):
            coulomb[i, i] = 4*math.pi/(q_vec[i]**2+q_p**2)
    chi_to_inv = np.zeros((n_q, n_q), dtype = "c16")
    for i in range(n_w):
        for j in range(n_q):
            if chi0qgg[i, j, j] == 0:
                chi_out[i, j, j] = chi0qgg[i, j, j]
            chi_to_inv[j, j] = (chi0qgg[i, j, j])**(-1)-coulomb[j, j]
            chi_out[i, j, j] = chi_to_inv[j, j]**(-1)
    return chi_out

@jit(nopython = True, parallel=True)
def chi_jellium_slab_test1(chi0qgg, q_vec, q_p):
    """Computes the interacting density response function"""
    n_w, n_q = chi0qgg.shape[0], chi0qgg.shape[1]
    chi_out = np.zeros((n_w, n_q, n_q), dtype = "c16")
    coulomb = np.zeros((n_q, n_q))
    for i in range(n_q):
        if q_vec[i] == 0:
            coulomb[i, i] = 0
        else:
            coulomb[i, i] = 4*math.pi/(q_vec[i]**2+q_p**2)         
    for i in range(n_w):
        chi_out[i] = np.linalg.inv(np.linalg.inv(chi0qgg[i])-coulomb)
    return chi_out

@jit(nopython = True, parallel=True)
def chi0wzz_slab_jellium_with_pot_no_numba(q_p, v_pot, omega, dens, d_sys, eta):
    """Computes the density response function as found by Eguiluz with the slab represented as an infinite well"""
    n_w = len(omega)
    n_z = len(v_pot)
    z_vec = np.linspace(0, d_sys, n_z)
    chi0wzz = np.zeros((n_w, math.ceil(n_z/2), n_z), dtype = "c16")
    energies, bands = ss.eig_energie(v_pot, z_vec)
    index = list(np.argsort(energies))
    bands_sorted = bands[:, index]
    energies = (energies[index])
    e_f, nmax = ef_2D_full(dens, d_sys, energies)
    bands_z = np.zeros((bands.shape), dtype = "c16")
    for i in range(n_z):
        bands_z[:, i] = np.fft.fft((bands_sorted[:, i]))
    wff = np.zeros((n_z, n_z, n_z), dtype = "c16")
    for i in range(n_z):
        for j in range(n_z):
            for k in range(n_z):
                wff[i, j, k] = bands_z[i, k]*bands_z[j,k]
    for i in range(n_w):
        fll = np.zeros((nmax, n_z), dtype = "c16")
        for j in range(nmax):
            for k in range(n_z):
                fll[j, k] = f_ll_pot(q_p, omega[i], energies[j], energies[k], e_f, eta)
        for j in range(math.ceil(n_z/2)):
            for k in range(n_z):
                for l in range(nmax):
                    wffi = wff[j, k, l]
                    for m in range(round(n_z*2/3)):
                        wffj = wff[j, k, m]
                        chi0wzz[i, j, k]+=wffi*wffj*fll[l, m]  
    return chi0wzz/n_z**(3/2), energies

@jit(nopython = True, parallel=True)
def a_ll_pot_no_numba(q_p, e_l1, e_l2):
    """Segment of the prefactor of Eguiluz1985"""
    return (q_p**2)/2-(e_l1-e_l2)


@jit(nopython = True, parallel=True)
def f_ll_pot_no_numba(q_p, omega, e_l1, e_l2, e_f, eta):
    """Compute the prefactor from the formula of Eguiluz1985"""
    a_l1l2 = a_ll_pot(q_p, e_l1, e_l2)
    if q_p == 0:
        pre_factor = (e_f-e_l1)/(math.pi)
        return -pre_factor*(1/(a_l1l2+omega+ic*eta)+1/(a_l1l2-omega-ic*eta))
    else:
        k_l = cmath.sqrt(2*(e_f-e_l1))
        return -1/(math.pi*q_p**2)*(2*a_l1l2+ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2-omega-ic*eta)**2)-ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2+omega+ic*eta)**2))


def pre_run_chi0(v_pot, z_vec, dens, d_sys):
    n_z = len(v_pot)
    energies, bands = ss.eig_energie(v_pot, z_vec)
    index = list(np.argsort(energies))
    bands_sorted = bands[:, index]
    energies = (energies[index])
    #energies = np.append(np.array([0]),energies)
    e_f, nmax = ef_2D_full(dens, d_sys, energies)
    bands_z = np.zeros((bands.shape), dtype = "c16")
    for i in range(n_z):
        bands_z[:, i] = np.fft.ifft((bands_sorted[:, i]))
        bands_z[:, i] = bands_z[:, i]/np.linalg.norm(bands_z[:, i])
    return energies, bands_z, e_f, nmax

@jit(debug = True, nopython = True, parallel=True)
def chi0wzz_slab_jellium_with_pot(q_p, energies, bands_z, omega, e_f, nmax, d_sys, eta, sym = True):
    """Computes the density response function as found by Eguiluz with the slab represented as an infinite well"""
    #print("Computation from potential started")
    n_w = len(omega)
    #energies = energies[1::]
    #nmax = nmax-1
    n_z = len(energies)
    
    wff = np.zeros((n_z, n_z, n_z), dtype = "c16")
    #wff2 = np.zeros((n_z, n_z, n_z), dtype = "c16")
    for i in range(n_z):
        for j in range(n_z):
            for k in range(n_z):
                wff[i, j, k] = bands_z[i, k]*np.conj(bands_z[j,k])
                #wff1[i, j, k] = np.conj(bands_z[i, k])*bands_z[j,k]
                #wff2[i, j, k] = bands_z[i, k]*np.conj(bands_z[j,k])
    nband_tot = 4*nmax
    if sym:
        n_half = math.ceil(n_z/2)
    else:
        n_half = n_z
    chi0wzz = np.zeros((n_w, n_half, n_z), dtype = "c16")
    for i in range(n_w):
        fll = np.zeros((nmax, nband_tot), dtype = "c16")
        for j in range(nmax):
            for k in range(4*nmax):
                fll[j, k] = f_ll_pot(q_p, omega[i], energies[j], energies[k], e_f, eta)
        for j in range(n_half):
            for k in range(n_z):
                for l in range(nmax):
                    wffi = np.conj(wff[j, k, l])
                    #wffi2 = wff2[j, k, l]
                    for m in range(nband_tot):
                        wffj = wff[j, k, m]
                        #wffj2 = np.conj(wff2[j, k, m])
                        chi0wzz[i, j, k]+=(wffi*wffj)*fll[l, m]
                        #chi0wzz[i, j, k]+=(wffi1*wffj1+wffi2*wffj2)*fll[l, m]/2 
    return chi0wzz*n_z**2/d_sys**2

@jit(debug = True, nopython = True)
def a_ll_pot(q_p, e_l1, e_l2):
    """Segment of the prefactor of Eguiluz1985"""
    return (q_p**2)/2-(e_l1-e_l2)


@jit(debug = True, nopython = True)
def f_ll_pot(q_p, omega, e_l1, e_l2, e_f, eta):
    """Compute the prefactor from the formula of Eguiluz1985"""
    a_l1l2 = a_ll_pot(q_p, e_l1, e_l2)
    if q_p == 0:
        pre_factor = (e_f-e_l1)/(math.pi)
        return -pre_factor*(1/(a_l1l2+omega+ic*eta)+1/(a_l1l2-omega-ic*eta))
    else:
        k_l = cmath.sqrt(2*(e_f-e_l1))
        return -1/(math.pi*q_p**2)*(2*a_l1l2+ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2-omega-ic*eta)**2)-ic*cmath.sqrt(q_p**2*k_l**2-(a_l1l2+omega+ic*eta)**2))