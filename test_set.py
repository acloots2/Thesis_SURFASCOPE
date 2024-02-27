"""Set of functions to obtain test the validity of the implementation
    of the different functions to obtain the dielectric properties"""

import numpy as np
import jellium_slab as js
import tools
import numba
import math
import cmath
import potentials as pot
from numba import jit


def weight_majerus(eps_wgg):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    for i in range(len(vec_p[0, :])):
        vec_p[:, i] = vec_p[i]    
    vec_dual_p = np.linalg.inv(vec_p)
    vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
    vec_dual = vec_dual_p
    vec = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    for i in range(1, 2):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        vec_dual_p = np.linalg.inv(vec_p)
        print(np.linalg.norm(vec_p[:, 0]), np.linalg.norm(vec_dual_p[0, :]))
        ####
        vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
        overlap = np.abs(np.dot(vec_dual, vec_p))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < n_q:  # add missing indices
            addlist = []
            removelist = []
            for j in range(n_q):
                if index.count(j) < 1:
                    addlist.append(j)
                if index.count(j) > 1:
                    for l in range(1, index.count(j)):
                        removelist+= list(np.argwhere(np.array(index) == j)[l])
            for j in range(len(addlist)):
                index[removelist[j]] = addlist[j]
        ####

        vec = vec_p[:, index]
        vec_dual = vec_dual_p[index, :]
        eig[i] = eig_all[i, index]
        weights_p[i]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    return weights_p, eig

def weight_majerus_diff(eps_wgg):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    vec_dual_p = np.linalg.inv(vec_p)
    vec_dual = vec_dual_p
    vec = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    for i in range(1, n_w):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        vec_dual_p = np.linalg.inv(vec_p)
        ####
        overlap = np.abs(np.sum(vec_p-vec[i-1]))
        #overlap = np.abs(np.dot(vec_dual, vec_p))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < n_q:  # add missing indices
            addlist = []
            removelist = []
            for j in range(n_q):
                if index.count(j) < 1:
                    addlist.append(j)
                if index.count(j) > 1:
                    for l in range(1, index.count(j)):
                        removelist+= list(np.argwhere(np.array(index) == j)[l])
            for j in range(len(addlist)):
                index[removelist[j]] = addlist[j]
        ####

        vec = vec_p[:, index]
        vec_dual = vec_dual_p[index, :]
        eig[i] = eig_all[i, index]
        weights_p[i]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    return weights_p, eig

@jit(nopython = True, parallel=True)
def weight_full_pause(eps_wgg):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec_dual[0] = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    vec_dual_p = np.linalg.inv(vec_p)
    """for i in range(n_q):
        vec_dual_p[i, :] = vec_dual_p[i, :]/np.linalg.norm(vec_dual_p[i, :])"""
    #vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
    vec_dual[0] = vec_dual_p
    vec[0] = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0, 0,:],(np.transpose(vec_dual[0, :,0])))
    for i in range(1, n_w):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        vec_dual_p = np.linalg.inv(vec_p)
        """for j in range(n_q):
            vec_dual_p[j, :] = vec_dual_p[j, :]/np.linalg.norm(vec_dual_p[j, :])"""
        ####
        overlap = np.abs(np.dot(vec_dual[i-1], vec_p))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < n_q:  # add missing indices
            addlist = []
            removelist = []
            for j in range(n_q):
                if index.count(j) < 1:
                    addlist.append(j)
                if index.count(j) > 1:
                    for l in range(1, index.count(j)):
                        removelist+= list(np.argwhere(np.array(index) == j)[l])
            for j in range(len(addlist)):
                index[removelist[j]] = addlist[j]
        ####

        vec[i] = vec_p[:, index]
        vec_dual[i] = vec_dual_p[index, :]
        eig[i] = eig_all[i, index]
        weights_p[i]=np.multiply(vec[i, 0,:],(np.transpose(vec_dual[i, :,0])))
    return weights_p, eig, vec, vec_dual

#@jit(nopython = True, parallel=True)
def weight_full_test(eps_wgg, q_vec):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec_dual[0] = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    for i in range(n_q):
        vec_p[:, i] = vec_p[:, i]/tools.func_norm(q_vec, vec_p[:, i])
    vec_dual_p = np.linalg.inv(vec_p)
    for i in range(n_q):
        vec_dual_p[i, :] = vec_dual_p[i, :]/tools.func_norm(q_vec, vec_dual_p[i, :])
    #vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
    vec_dual[0] = vec_dual_p
    vec[0] = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0, 0,:],(np.transpose(vec_dual[0, :,0])))
    for i in range(1, n_w):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        for j in range(n_q):
            vec_p[:, j] = vec_p[:, j]/tools.func_norm(q_vec, vec_p[:, j])
        vec_dual_p = np.linalg.inv(vec_p)
        for j in range(n_q):
            vec_dual_p[j, :] = vec_dual_p[j, :]/tools.func_norm(q_vec, vec_dual_p[j, :])
        ####
        vec[i] = vec_p
        vec_dual[i] = vec_dual_p
        eig[i] = eig_all[i]
        weights_p[i]=np.multiply(vec[i, 0,:],(np.transpose(vec_dual[i, :,0])))
    return weights_p, eig, vec, vec_dual


@jit(nopython = True, parallel=True)
def weight_full(eps_wgg):
    """Computes the weights as given in the thesis of Kirsten Andersen
    Code adapted from the GPAW software"""
    n_w, n_q = eps_wgg.shape[0], eps_wgg.shape[1]
    weights_p = np.zeros((n_w, n_q), dtype = "c16")
    eig_all = np.zeros((n_w, n_q), dtype = "c16")
    eig = np.zeros((n_w, n_q), dtype = "c16")
    vec_dual = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec = np.zeros((n_w, n_q, n_q), dtype = "c16")
    vec_dual[0] = np.diag(np.ones(n_q))
    eps_gg = eps_wgg[0]
    eig_all[0], vec_p = np.linalg.eig(eps_gg)
    vec_dual_p = np.linalg.inv(vec_p)
    """for i in range(n_q):
        vec_dual_p[i, :] = vec_dual_p[i, :]/np.linalg.norm(vec_dual_p[i, :])"""
    #vec_dual_p = vec_dual_p/np.linalg.norm(vec_dual_p[0, :])
    vec_dual[0] = vec_dual_p
    vec[0] = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0, 0,:],(np.transpose(vec_dual[0, :,0])))
    for i in range(1, n_w):
        eps_gg = eps_wgg[i]
        eig_all[i], vec_p = np.linalg.eig(eps_gg)
        vec_dual_p = np.linalg.inv(vec_p)
        """for j in range(n_q):
            vec_dual_p[j, :] = vec_dual_p[j, :]/np.linalg.norm(vec_dual_p[j, :])"""
        ####
        vec[i] = vec_p
        vec_dual[i] = vec_dual_p
        eig[i] = eig_all[i]
        weights_p[i]=np.multiply(vec[i, 0,:],(np.transpose(vec_dual[i, :,0])))
    return weights_p, eig, vec, vec_dual



@jit(nopython = True, parallel=True)
def loss_func_majerus(weights_p, eig):
    """Computes the loss function as given in the thesis of Bruno Majerus"""
    n_w, n_q = eig.shape
    loss_func = np.zeros((n_w), dtype = "c16")
    for i in range(n_q):
        loss_func_i = -np.imag(np.power(eig[:, i], -1))
        weight_i = weights_p[:, i]
        loss_func += np.multiply(loss_func_i, weight_i)
    return loss_func


def loss_full_slab_wov_test(diel, z_2, q_p):
    """Gives the spectra from the density response function"""
    diel_wov = js.sym_chi_slab(diel)
    q_vec = tools.zvec_to_qvec(z_2)
    diel_wov_q = js.fourier_inv(diel_wov, z_2)
    eps = js.epsilon(diel_wov_q, np.real(tools.inv_rev_vec(q_vec)), q_p)
    weights, eig_q, vec, vec_dual = weight_full(eps, q_vec)
    loss = loss_func_majerus(weights, eig_q)
    return loss, weights, eig_q, eps, vec, vec_dual

def loss_full_slab_wov(diel, z_2, q_p):
    """Gives the spectra from the density response function"""
    diel_wov = js.sym_chi_slab(diel)
    q_vec = tools.zvec_to_qvec(z_2)
    diel_wov_q = js.fourier_inv(diel_wov, z_2)
    eps = js.epsilon(diel_wov_q, np.real(tools.inv_rev_vec(q_vec)), q_p)
    weights, eig_q, vec, vec_dual = weight_full(eps)
    loss = loss_func_majerus(weights, eig_q)
    return loss, weights, eig_q, eps, vec, vec_dual


def loss_full_slab_wv(diel, z_2, q_p, void):
    """Gives the spectra from the density response function"""
    diel_wv = js.sym_chi_slab_with_void(diel, z_2, void)
    z_void = np.linspace(0, max(z_2)+void, len(diel_wv[0, 0, :]))
    q_vec = tools.zvec_to_qvec(z_void)
    diel_wv_q = js.fourier_inv(diel_wv, z_void)
    eps = js.epsilon(diel_wv_q, np.real(tools.inv_rev_vec(q_vec)), q_p)
    weights, eig_q = weight_majerus(eps)
    #weights = np.real(weights)-np.min(np.real(weights))
    loss = loss_func_majerus(weights, eig_q)
    return loss, weights, eig_q

@jit(nopython = True, parallel=True)
def mode_filter(eig_r):
    n_w, n_q = eig_r.shape[0], eig_r.shape[1]
    average = np.zeros((n_w))
    for i in range(n_w):
        average[i]  = np.sum(np.abs(np.real(eig_r[i, 0:round(n_q/4), :])))/(np.round(n_q/4)*n_q)
    surf_mode = {}
    for i in range(n_w):
        surf_mode[i] = []
        for j in range(n_q):
            if np.sum(np.abs(np.real(eig_r[i, 0:11, j])))/11 < average[i]/2:
                surf_mode[i].append(j)
    return surf_mode   

def d_wall(k_f, l):
    return l/2+1/(8*k_f)*(3*math.pi+cmath.sqrt(16*(k_f*l)**2+24*math.pi*k_f*l+25*math.pi**2))

def d_slab(k_f, d_w):
    return d_w - 2*(1/(8*k_f)*(3*math.pi+math.pi**2/(k_f*d_w)))


def system_optimizer(d_slab_start, dens_start, d_void, point_dens = 1):
    d_slab_inf = 1000
    #d_void = 250
    well_depth = 1000
    v_pot, z_pot = pot.square_well_pot(d_slab_inf, d_void, well_depth, point_dens)
    energies, bands, e_f_inf, nmax = js.pre_run_chi0(v_pot, z_pot, dens_start, d_slab_inf)
    d_inf = np.real(3*math.pi/(8*cmath.sqrt(2*e_f_inf)))
    d_well = d_slab_start+2*d_inf
    rho = dens_start*d_slab_start/d_well
    v_pot, z_pot = pot.square_well_pot(d_well, d_void, well_depth, point_dens)
    energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, rho, d_well)
    for i in range(5):
        d_well = np.real(d_wall(cmath.sqrt(2*e_f), d_slab_start))
        rho = dens_start*d_slab_start/d_well
        v_pot, z_pot = pot.square_well_pot(d_well, d_void, well_depth, point_dens)
        energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, rho, d_well)
    print(d_well, rho, e_f)
    return v_pot, z_pot, energies, bands, e_f, nmax, rho, d_well

