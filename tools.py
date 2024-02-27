"""Set of functions that are useful to manipulate vectors for figures or Fourier Transform"""

import math
import cmath
import numpy as np
import numba
from numba import jit

@jit(nopython = True)
def qvec_to_zvec(q_vec):
    """Enter the q vector (-q, q) with the point q=0 so always an odd number of point"""
    npnt = len(q_vec)
    center = math.floor(npnt/2)+1
    qmin = q_vec[center]
    return np.linspace(0, 2*math.pi/qmin, npnt)

@jit(nopython = True)
def zvec_to_qvec(z_vec):
    """enter the z vector (0, zmax)"""
    npnt = len(z_vec)
    z_sort = np.sort(np.abs(z_vec))
    if z_sort[0] != 0:
        raise ValueError("Watch out, the given vector does not contain z = 0 which is essential in this implementation to obtain the coreect vector in reciprocal space")
    zmin = z_sort[1]
    qnosym = np.linspace(-math.pi/zmin, math.pi/zmin, npnt)
    qsym = np.zeros((npnt))
    for i in range(math.floor(npnt/2)+1):
        q_vec = (-qnosym[i]+qnosym[npnt-i-1])/2
        qsym[i] = -q_vec
        qsym[npnt-i-1] = q_vec
    return qsym

@jit(nopython = True)
def center_z(z_vec):
    """Set the origin to the center of the slab rather than on the far left of the computed area"""
    npnt = len(z_vec)
    z_half = max(z_vec)/2
    return np.linspace(-z_half, z_half, npnt)

@jit(nopython = True)
def rev_vec(y_in):
    """In : [0, 1, 2, 3, 3, 2, 1, 0] // [0, 1, 2, 3, 3, 2, 1]
       Out : [3, 2, 1, 0, 0, 1, 2, 3] // [3, 2, 1, 0, 1, 2, 3]"""
    npnt = y_in.size
    y_out = np.zeros((npnt), dtype = "c16")
    if npnt%2==0:
        mid = math.floor(npnt/2)
    else:
        mid= math.floor(npnt/2)+1
    y_out[0:mid] = y_in[0:mid][::-1]
    y_out[mid:npnt] = y_in[mid:npnt][::-1]
    return y_out

@jit(nopython = True)
def inv_rev_vec(y_in):
    """In : [-3, -2, -1, 0, 0, 1, 2, 3] // [-3, -2, -1, 0, 1, 2, 3]
       Out : [0, 1, 2, 3, -3, -2, -1, -0] // [0, 1, 2, 3, -3, -2, -1]"""
    npnt = y_in.size
    y_out = np.zeros((npnt), dtype = "c16")
    if npnt%2==0:
        mid = math.floor(npnt/2)
        y_out[0:mid] = y_in[mid:npnt]
        y_out[mid:npnt] = y_in[0:mid]
    else:
        mid = math.floor(npnt/2)+1
        y_out[0:mid] = y_in[mid-1:npnt]
        y_out[mid:npnt] = y_in[0:mid-1]
    return y_out

@jit(nopython = True)
def extract_diag(matwdd):
    """"Extracts the diagonal of 3D matrix by extracting its diagonals"""
    n_w, n_d = matwdd.shape[0], matwdd.shape[1]
    mat_out = np.zeros((n_w, n_d), dtype = "c16")
    for i in range(n_w):
        mat_out[i] = np.diag(matwdd[i])
    return mat_out

@jit(nopython = True)
def to_diag(matwdd):
    """Changes a 2D Matrix into a 3D one by placing all its values on the diagonals"""
    n_w, n_d = matwdd.shape[0], matwdd.shape[1]
    mat_out = np.zeros((n_w, n_d, n_d), dtype = "c16")
    for i in range(n_w):
        mat_out[i] = np.diag(matwdd[i])
    return mat_out

@jit(nopython = True)
def parity(vec):
    """Test the parity of a given eigen vector"""
    if len(vec)<=6:
        raise ValueError("The vector is too short")
    vec_inv = vec[::-1]
    i_test = 3
    ratio = np.zeros((i_test), dtype = "c16")
    for i in range(1, i_test+1):
        ratio[i-1] = vec[i]/vec_inv[i]
    #print(ratio)
    parity_test = round(np.sum(ratio)/i_test, 3)
    if parity_test == 1:
        return "even"
    elif parity_test == -1:
        return "odd"
    else:
        return "void"

@jit(nopython = True)
def parity_perm_old(vec):
    """Test the parity of a given eigen vector"""
    if len(vec)<=6:
        raise ValueError("The vector is too short")
    vec_inv = vec[::-1]
    i_test = 3
    ratio = np.zeros((i_test), dtype = "c16")
    for i in range(1, i_test+1):
        ratio[i-1] = vec[i+10]/vec_inv[i+9]
    #print(ratio)
    parity_test = np.sum(np.real(ratio))/i_test
    parity_test = round(parity_test, 3)
    #print(parity_test)
    if parity_test == 1:
        return "even"
    elif parity_test == -1:
        return "odd"
    else:
        return "void"

@jit(nopython = True)
def parity_perm(vec):
    """Test the parity of a given eigen vector"""
    if len(vec)<=6:
        raise ValueError("The vector is too short")
    sum_test = np.real(np.round(np.sum(vec), 8))
    #Íprint(sum_test)
    if sum_test == 0:
        return "odd"
    else:
        return "even"
    
def density(energies, bands, e_f, point_dens = 1):
    index = 0
    npnt = len(energies)
    dens = np.zeros((npnt), dtype = complex)
    while energies[index] < e_f:
        dens += (e_f-energies[index])*bands[:, index]*np.conj(bands[:, index])
        index += 1
    return 1/math.pi*dens

@jit(nopython = True, parallel=True)
def func_norm(x, fx):
    return cmath.sqrt(np.sum(np.multiply(np.conj(fx), fx))*np.abs(x[0]-x[1]))

def fourier_dir(matwgg):
    """Performs the Fourier Transform to go from chi0wqq to chi0wzz"""
    n_w, nq1, nq2 = matwgg.shape
    matwzq2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq1):
            matwzq2[i, j, :] = np.fft.ifft(matwgg[i, j, :])
    matwz1z2 = np.zeros((n_w, nq1, nq2), dtype = "c16")
    for i in range(n_w):
        for j in range(nq2):
            matwz1z2[i, :, j] = np.fft.ifft(matwzq2[i, :, j])
    matwz1z2 = matwz1z2*nq2
    matwz1z2_out = np.zeros((n_w, nq1, nq2), dtype = "c16")    
    for i in range(n_w):
        matwz1z2_out[i] = (matwz1z2[i, :, :]+np.transpose(matwz1z2[i, :, :]))/2
    return matwz1z2_out

@jit(nopython = True)
def second_derivative(v_p, h_step):
    """Computes the second derivative a v_p"""
    n_q = len(v_p)
    s_d = np.zeros((n_q), dtype = "c16")
    for i in range(1, n_q-1):
        s_d[i] = v_p[i-1]-2*v_p[i]+v_p[i+1]
    return s_d/h_step**2

@jit(nopython = True)
def coulomb(q_vec, q_p):
    """Computes the Coulomb potential with V_c[0+q_p]=0"""
    n_q = len(q_vec)
    coulomb_vec = np.zeros((n_q, n_q))
    if q_p ==0:
        for i in range(n_q):
            if q_vec[i] == 0:
                coulomb_vec[i, i] = 0
            else:
                coulomb_vec[i, i] = q_vec[i]**(-2)
    else:
        coulomb_vec = np.power(np.power(q_vec, 2)+q_p**2, -1)
    return coulomb_vec

def shift(chi_0):
    nw, nz = chi_0.shape[0], chi_0.shape[1]
    nz_half = round(math.ceil(nz/2))
    chi0_shift1 = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        for j in range(nz):
            chi0_shift1[i, j, 0:nz_half] = chi_0[i, j, -nz_half::]
            chi0_shift1[i, j, -nz_half+1::] = chi_0[i, j, 0:nz_half-1]
    chi0_shift2 = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        for j in range(nz):
            chi0_shift2[i, 0:nz_half, j] = chi0_shift1[i, -nz_half::, j]
            chi0_shift2[i, -nz_half+1::, j] = chi0_shift1[i, 0:nz_half-1, j]
    return chi0_shift2

def shift_c(coulomb_2d):
    nz = coulomb.shape[0]
    nz_half = round(math.ceil(nz/2))
    coulomb_shift1 = np.zeros((nz, nz), dtype = complex)
    for j in range(nz):
        coulomb_shift1[j, 0:nz_half] = coulomb_2d[j, -nz_half::]
        coulomb_shift1[j, -nz_half+1::] = coulomb_2d[j, 0:nz_half-1]
    coulomb_shift2 = np.zeros((nz, nz), dtype = complex)
    for j in range(nz):
        coulomb_shift2[0:nz_half, j] = coulomb_shift1[-nz_half::, j]
        coulomb_shift2[-nz_half+1::, j] = coulomb_shift1[0:nz_half-1, j]
    return coulomb_shift2