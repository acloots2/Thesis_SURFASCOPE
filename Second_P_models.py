from asyncio import constants
from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
import time 
from abipy.electrons.scr import ScrFile
import numpy as np
import cmath
import math
import scipy
import pointcloud as pc
import DRF
import plotly.graph_objects as go
import abipy
from scipy.interpolate import RegularGridInterpolator

ic = complex(0,1)
e0 = 1/(4*math.pi)

def im_chi0_XG(q, omega, n=0.025):
    npnt = len(q)
    nw = len(omega)
    chi0q = np.zeros((nw, npnt), dtype = complex)
    E_F = (1/2)*(3*math.pi**2*n)**(2/3)
    for i in range(nw):
        w = omega[i]
        for j in range(npnt):
            q_norm = abs(q[j])
            if q_norm==0:
                continue
            E_minus = (w-(q_norm**2)/2)**2*(2/(q_norm**2))*1/4
            if E_minus<=(E_F-w):
                chi0q[i, j]=1/(2*math.pi)*(1/q_norm)*w
            elif E_F >= E_minus >= E_F-w:
                chi0q[i, j]=1/(2*math.pi)*(1/q_norm)*(E_F-E_minus)
            else:
                continue
    return -chi0q

def re_chi0_XG(q, omega, n=0.025):
    npnt = len(q)
    nw = len(omega)
    chi0q = np.zeros((nw, npnt), dtype = complex)
    E_F = (1/2)*(3*math.pi**2*n)**(2/3)
    k_F = (3*math.pi**2*n)**(1/3)
    pref = -4/(2*math.pi)**2*k_F
    for i in range(nw):
        w = omega[i]
        k_w = (2*w)**(1/2)
        for j in range(npnt):
            q_norm = abs(q[j])
            if q_norm==0:
                continue
            else:
                t1 = 1-(1/4)*((k_w**2-q_norm**2)**2/(k_F**2*q_norm**2))
                t21 = (k_w**2-2*k_F*q_norm-q_norm**2)/(k_w**2+2*k_F*q_norm-q_norm**2)
                t22 = math.log(abs(t21))
                t3 = 1-(1/4)*((k_w**2+q_norm**2)**2/(k_F**2*q_norm**2))
                t41 = (k_w**2+2*k_F*q_norm+q_norm**2)/(k_w**2-2*k_F*q_norm+q_norm**2)
                t42 = math.log(abs(t41))
                chi0q[i, j] = 1/2+(k_F/(4*q_norm))*(t1*t22+t3*t42)
    return pref*chi0q

def chi0q_XG(q, omega, n = 0.025):
    rchi = re_chi0_XG(q, omega, n)
    ichi = im_chi0_XG(q, omega, n)
    chi0q = rchi+ic*ichi
    return chi0q

def chiq(chi0q, q, q_p, opt = "positive"):
    #Option are "positive" with q values from 0 to q_max or "symmetric" with values from -q_max to q_max
    nw, nq = chi0q.shape
    chiq_m = np.zeros((nw, nq), dtype = complex)
    coulomb = np.zeros((nq))
    if opt == "positive" and q_p != 0:
        coulomb = 4*math.pi*np.power(np.abs(q+q_p), -2)
    elif opt == "positive" and q_p == 0:
        coulomb[1:nq] = 4*math.pi*np.power(np.abs(q[1:nq]), -2)
    elif opt == "symmetric" and q_p == 0:
        q = np.real(Inv_Rev_vec(q))
        coulomb[1:nq] = 4*math.pi*np.power(np.abs(q[1:nq]), -2)
    else:
        q = np.real(Inv_Rev_vec(q))
        coulomb = 4*math.pi*np.power(np.abs(q+q_p), -2)
    for i in range(nw):
        for j in range(1, nq):
            chiq_inv = np.power(chi0q[i, j], -1, dtype=complex)-coulomb[j]
            chiq_m[i, j] = np.power(chiq_inv, -1, dtype=complex)
    return chiq_m

def chiq_mat(chi0q, q, q_p, d, n=0.025):
    nw, nq, nq0 = chi0q.shape
    chiq_m = np.zeros((nw, nq, nq), dtype = complex)
    coulomb = np.zeros((nq, nq))
    if q_p == 0:
        for i in range(1, nq):
            coulomb[i, i] = 4*math.pi*np.power((q[i]), -2)*(1-math.cos(q[i]*d/2))
    else:
        for i in range(nq):
            coulomb[i, i] = 4*math.pi/(q[i]**2+q_p**2)*(1-math.cos(q[i]*d/2))
    #print(np.diag(coulomb))
    chi_to_inv = np.zeros((nq, nq), dtype = complex)
    for i in range(nw):
        chi_to_inv = np.linalg.inv(chi0q[i])-coulomb
        chiq_m[i] = np.linalg.inv(chi_to_inv)
    return chiq_m

def epsilon_Wilson(chi0qGG, q, q_p, opt = "Slab"):
    if opt == "Slab":
        nw, nq, nq0 = chi0qGG.shape
        eps_out = np.zeros((nw, nq, nq), dtype = complex)
        coulomb = np.zeros((nq, nq))
        if q_p == 0:
            for i in range(1, nq):
                if q[i]==0:
                    continue
                coulomb[i, i] = 4*math.pi*np.power((q[i]), -2)#*(1-math.cos(q[i]*d/2)) 
        else:
            for i in range(nq):
                coulomb[i, i] = 4*math.pi/(q[i]**2+q_p**2)#*(1-math.cos(q[i]*d/2))
        for i in range(nw):
            eps_out[i] = np.diag(np.ones(nq))-np.matmul(coulomb, chi0qGG[i])
    elif opt == "Bulk":
        nw, nq= chi0qGG.shape
        eps_out = np.zeros((nw, nq), dtype = complex)
        coulomb = np.zeros((nq))
        for i in range(nq):
            if q[i]==0:
                continue
            coulomb[i] = 4*math.pi/(q[i]**2)#*(1-math.cos(q[i]*d/2))
        for i in range(nw):
            eps_out[i] = np.ones(nq)-np.multiply(coulomb, chi0qGG[i])
    else:
        raise ValueError("The specified option does not exist")
    return eps_out

def epsilon_Wilson_test_bulk(chi0qGG, q, q_p, opt = "Slab"):
    if opt == "Slab":
        nw, nq, nq0 = chi0qGG.shape
        eps_out = np.zeros((nw, nq, nq), dtype = complex)
        coulomb = np.zeros((nq, nq))
        mid = round(nq/2)
        print(mid)
        if q_p == 0:
            for i in range(0, mid):
                coulomb[i, i] = 4*math.pi*np.power((q[i]), -2)#*(1-math.cos(q[i]*d/2)) 
            for i in range(mid+1, nq):
                coulomb[i, i] = 4*math.pi*np.power((q[i]), -2)
        else:
            for i in range(nq):
                coulomb[i, i] = 4*math.pi/(q[i]**2+q_p**2)#*(1-math.cos(q[i]*d/2))
        for i in range(nw):
            eps_out[i] = np.diag(np.ones(nq))-np.matmul(coulomb, chi0qGG[i])
    elif opt == "Bulk":
        nw, nq= chi0qGG.shape
        eps_out = np.zeros((nw, nq), dtype = complex)
        coulomb = np.zeros((nq))
        for i in range(1, nq):
                coulomb[i] = 4*math.pi/(q[i]**2)#*(1-math.cos(q[i]*d/2))
        for i in range(nw):
            eps_out[i] = np.ones(nq)-np.matmul(coulomb, chi0qGG[i])
    else:
        raise ValueError("The specified option does not exist")
    return eps_out


def eig_plasmons(eps):
    nw, nq, nq0 = eps.shape
    eig_value = np.zeros((nw, nq), dtype = complex)
    eig_l_vector = np.zeros((nw, nq, nq), dtype =complex)
    eig_r_vector = np.zeros((nw, nq, nq), dtype =complex)
    for i in range(nw):
        eig_value[i], eig_l_vector[i], eig_r_vector[i] = scipy.linalg.eig(eps[i], left= True)
        #eig_value[i] = sorted(eig_value[i])
    return eig_value, eig_l_vector, eig_r_vector

def np_eig_plasmons(eps):
    nw, nq, nq0 = eps.shape
    eig_value = np.zeros((nw, nq), dtype = complex)
    eig_r_vector = np.zeros((nw, nq, nq), dtype =complex)
    for i in range(nw):
        eig_value[i], eig_r_vector[i] = np.linalg.eig(eps[i])
        #eig_value[i] = sorted(eig_value[i])
    return eig_value, eig_r_vector

def inv_eig_plasmons(eps_eig):
    nw, nq = eps_eig.shape
    inv_eig_v = np.zeros((nw, nq), dtype = complex)
    for i in range(nw):
        for j in range(nq):
            inv_eig_v[i, j] = np.power(eps_eig[i, j], -1)
    
    return inv_eig_v

def weights(eig_r, eig_l):
    nw, nq, nq0 = eig_r.shape
    weight = np.zeros((nw, nq), dtype = complex)
    delta = np.zeros((nw, nq), dtype = complex)
    for i in range(nw):
        for j in range(nq):
            for k in range(nq):
                delta[i, j] += np.conj(eig_l[i, k, j])*eig_r[i, k, j]
    for i in range(nw):
        for j in range(nq):
            weight[i, j] = eig_r[i, 0, j]*np.conj(eig_l[i, 0, j])/delta[i, j]
    return weight

def weight_Majerus(eps_wGG):
    nw, nq, nq0 = eps_wGG.shape
    weights_p = np.zeros((nw, nq), dtype = complex)
    eig_all = np.zeros((nw, nq), dtype = complex)
    eig = np.zeros((nw, nq), dtype = complex)
    vec_dual = np.diag(np.ones(nq))
    eps_GG = eps_wGG[0]
    eig_all[0], vec_p = np.linalg.eig(eps_GG)
    vec_dual_p = np.linalg.inv(vec_p)
    vec_dual = vec_dual_p
    vec = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    for i in range(1, nw):
        eps_GG = eps_wGG[i]
        eig_all[i], vec_p = np.linalg.eig(eps_GG)
        vec_dual_p = np.linalg.inv(vec_p)
        ####
        overlap = np.abs(np.dot(vec_dual, vec_p))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < nq:  # add missing indices
            addlist = []
            removelist = []
            for j in range(nq):
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
    return weights_p, eig, eig_all


def weight_Majerus_bulk(eps_wGG):
    nw, nq = eps_wGG.shape
    weights_p = np.zeros((nw, nq), dtype = complex)
    eig_all = np.zeros((nw, nq), dtype = complex)
    eig = np.zeros((nw, nq), dtype = complex)
    vec_dual = np.diag(np.ones(nq))
    eps_GG = eps_wGG[0]
    eig_all[0], vec_p = np.linalg.eig(np.diag(eps_GG))
    vec_dual_p = np.linalg.inv(vec_p)
    vec_dual = vec_dual_p
    vec = vec_p
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    for i in range(1, nw):
        eps_GG = eps_wGG[i]
        eig_all[i], vec_p = np.linalg.eig(np.diag(eps_GG))
        vec_dual_p = np.linalg.inv(vec_p)
        ####
        overlap = np.abs(np.dot(vec_dual, vec_p))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < nq:  # add missing indices
            addlist = []
            removelist = []
            for j in range(nq):
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
    return weights_p, eig, eig_all


def weight_Majerus_test(eps_wGG):
    nw, nq, nq0 = eps_wGG.shape
    weights_p = np.zeros((nw, nq), dtype = complex)
    eig_all = np.zeros((nw, nq), dtype = complex)
    eig = np.zeros((nw, nq), dtype = complex)
    vec_dual = np.diag(np.ones(nq))
    eps_GG = eps_wGG[0]
    eig_all[0], vec_p_l, vec_p_r = scipy.linalg.eig(eps_GG, left=True)
    vec_dual_p = vec_p_l
    vec_dual = vec_dual_p
    vec = vec_p_r
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.conj(vec_dual[0,:])))
    for i in range(1, nw):
        eps_GG = eps_wGG[i]
        eig_all[i], vec_p_l, vec_p_r = scipy.linalg.eig(eps_GG, left=True)
        vec_dual_p = vec_p_l
        ####
        overlap = np.abs(np.dot(np.conj(vec_dual), vec_p_r))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < nq:  # add missing indices
            addlist = []
            removelist = []
            for j in range(nq):
                if index.count(j) < 1:
                    addlist.append(j)
                if index.count(j) > 1:
                    for l in range(1, index.count(j)):
                        removelist+= list(np.argwhere(np.array(index) == j)[l])
            for j in range(len(addlist)):
                index[removelist[j]] = addlist[j]
        ####

        vec = vec_p_r[:, index]
        vec_dual = vec_dual_p[:, index]
        eig[i] = eig_all[i, index]
        delta = np.zeros((nq), dtype = complex)
        vec_inv_l = np.linalg.inv(vec_dual)
        vec_inv_r = np.linalg.inv(vec)
        for j in range(nq):
            for k in range(nq):
                delta[j] += np.conj(vec_inv_l[k, j])*vec_inv_r[k, j]
        weights_p[i]=np.divide(np.multiply(vec[0,:],(np.conj(vec_dual[0,:]))),delta)
    return weights_p, eig, eig_all

def weight_Majerus_test_1(eps_wGG):
    nw, nq, nq0 = eps_wGG.shape
    weights_p = np.zeros((nw, nq), dtype = complex)
    eig_all = np.zeros((nw, nq), dtype = complex)
    eig = np.zeros((nw, nq), dtype = complex)
    vec_p_l = np.zeros((nw, nq, nq), dtype = complex)
    vec_p_r = np.zeros((nw, nq, nq), dtype = complex)
    delta = np.zeros((nw, nq), dtype = complex)
    for i in range(nw):
        eig_all[i], vec_p_l[i], vec_p_r[i] = scipy.linalg.eig(eps_wGG[i], left=True)

    for i in range(nw):
        for j in range(nq):
            for k in range(nq):
                delta[i, j] += np.conj(vec_p_l[i, k, j])*vec_p_r[i, k, j]
    for i in range(nw):
        for j in range(nq):
            vec_p_r[i, j, :] = vec_p_r[i, j, :]*delta[i,j]
    vec_dual = np.diag(np.ones(nq))
    eps_GG = eps_wGG[0]
    eig_all[0], vec_p_l, vec_p_r = scipy.linalg.eig(eps_GG, left=True)
    vec_dual_p = vec_p_l
    vec_dual = vec_dual_p
    vec = vec_p_r
    eig[0] = eig_all[0, :]
    weights_p[0]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    for i in range(1, nw):
        eps_GG = eps_wGG[i]
        eig_all[i], vec_p_l, vec_p_r = scipy.linalg.eig(eps_GG, left=True)
        vec_dual_p = vec_p_l
        ####
        overlap = np.abs(np.dot(np.transpose(vec_dual), vec_p_r))
        index = list(np.argsort(overlap)[:, -1])
        if len(np.unique(index)) < nq:  # add missing indices
            addlist = []
            removelist = []
            for j in range(nq):
                if index.count(j) < 1:
                    addlist.append(j)
                if index.count(j) > 1:
                    for l in range(1, index.count(j)):
                        removelist+= list(np.argwhere(np.array(index) == j)[l])
            for j in range(len(addlist)):
                index[removelist[j]] = addlist[j]
        ####

        vec = vec_p_r[:, index]
        vec_dual = vec_dual_p[:, index]
        eig[i] = eig_all[i, index]
        weights_p[i]=np.multiply(vec[0,:],(np.transpose(vec_dual[:,0])))
    return weights_p, eig, eig_all

def Loss_Func_Majerus(eps_wGG):
    nw, nq = eps_wGG.shape[0], eps_wGG.shape[1]
    loss_func = np.zeros((nw), dtype = complex)
    weights_p, eig, eig_all = weight_Majerus(eps_wGG)
    for i in range(nq):
        loss_func_i = -np.imag(np.power(eig[:, i], -1))
        weight_i = weights_p[:, i]
        loss_func += np.multiply(loss_func_i, weight_i)
    return loss_func

def weights_test(eig_r, eig_l):
    nw, nq, nq0 = eig_r.shape
    weight = np.zeros((nw, nq), dtype = complex)
    delta = np.zeros((nw, nq), dtype = complex)
    half = round(nq/2)

    vec_inv_l = np.zeros((nw, nq, nq), dtype = complex)
    vec_inv_r = np.zeros((nw, nq, nq), dtype = complex)
    for i in range(nw):
        vec_inv_l[i] = np.linalg.inv(eig_l[i])
        vec_inv_r[i] = np.linalg.inv(eig_r[i])
        for j in range(nq):
            for k in range(nq):
                delta[i, j] += np.conj(vec_inv_l[i, k, j])*vec_inv_r[i, k, j]
    for i in range(nw):
        for j in range(nq):
            weight[i, j] =vec_inv_r[i, 0, j]*np.conj(vec_inv_l[i, 0, j])
    return weight

def Loss_Func(eig_v):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    for i in range(nq):
        loss_func_i = -np.imag(np.power(eig_v[:, i], -1))
        weight_i = np.ones(nw)
        loss_func += np.multiply(loss_func_i, weight_i)
    return loss_func

def Loss_Func_final(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    loss_func_i  = np.zeros((nw, nq), dtype = complex)
    weight = weights(eig_r, eig_l)
    weight_sum = np.zeros((nq), dtype = complex)
    for i in range(nq):
        loss_func_i[:, i] = -np.imag(np.power(eig_v[:, i], -1))
        weight_sum[i] = np.sum(weight[:, i])
    for i in range(nq):
        loss_func += loss_func_i[:, i]*weight_sum[i]
    return loss_func


def Loss_Func_test(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    loss_func_i  = np.zeros((nw, nq), dtype = complex)
    weight = weights_test(eig_r, eig_l)
    for i in range(nq):
        loss_func_i[:, i] = -np.imag(np.power(eig_v[:, i], -1))
    loss_func_weight = np.multiply(loss_func_i, weight)
    for i in range(nq):
        loss_func += np.sum(loss_func_weight[:, i])
    return loss_func

def Loss_Func_test0(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    loss_func_i  = np.zeros((nw, nq), dtype = complex)
    weight = weights_test(eig_r, eig_l)
    for i in range(nq):
        loss_func_i[:, i] = -np.imag(np.power(eig_v[:, i], -1))
    loss_func_weight = np.multiply(loss_func_i, weight)
    for i in range(nq):
        loss_func += np.sum(loss_func_weight[:, i])
    return loss_func



def Loss_Func_test_start(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    for i in range(nw):
        loss_func_i = -np.imag(np.power(eig_v[i, :], -1))
        weight_i = np.multiply(eig_r[i, 0, :], np.conj(eig_l[i, 0, :]))
        loss_func[i] = np.sum(np.multiply(loss_func_i, weight_i))
    return loss_func

def Loss_Func_test_end(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    for i in range(nw):
        loss_func_i = -np.imag(np.power(eig_v[i, :], -1))
        weight_i =np.abs(np.multiply(eig_r[i, :, 0], np.conj(eig_l[i, :, 0])))
        loss_func[i] = np.sum(np.multiply(loss_func_i, weight_i))
    return loss_func

def Loss_Func_test3(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    for i in range(nw):
        loss_func_i = -np.imag(np.power(eig_v[i, :], -1))
        weight_i =np.multiply(eig_r[i, :, 0], np.conj(eig_l[i, :, 0]))
        loss_func[i] = np.sum(np.multiply(loss_func_i, weight_i))
    return loss_func

def Loss_Func_test4(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    for i in range(nw):
        loss_func_i = -np.imag(np.power(eig_v[i, :], -1))
        weight_i =np.real(np.multiply(eig_r[i, :, 0], np.conj(eig_l[i, :, 0])))
        loss_func[i] = np.sum(np.multiply(loss_func_i, weight_i))
    return loss_func

def Loss_Func_test5(eig_v, eig_r, eig_l):
    nw, nq = eig_v.shape
    loss_func = np.zeros((nw), dtype = complex)
    weight_i = np.zeros((nw, nq), dtype = complex)
    for i in range(nw):
        loss_func_i = -np.imag(np.power(eig_v[i, :], -1))
        weight_i[i] =np.multiply(eig_r[i, :, 0], np.conj(eig_l[i, :, 0]))
    weight_i = weight_i-np.ones((nw, nq))*np.min(weight_i)
    for i in range(nw):
        loss_func[i] = np.sum(np.multiply(loss_func_i, weight_i[i]))
    return loss_func


def chiq_mat_test(chi0q, q, q_p, n=0.025):
    nw, nq, nq0 = chi0q.shape
    chiq_m = np.zeros((nw, nq, nq), dtype = complex)
    q_p2 = q_p**2
    coulomb = np.zeros((nq, nq))
    for i in range(1,nq):
        coulomb[i, i] = 4*math.pi*np.power(np.abs(q[i]+q_p), -2)
    coulomb[0, 0] = 0
    for i in range(nw):
        chiq_inv = np.zeros((nq, nq), dtype = complex)
        for j in range(nq):
            for k in range(nq):
                if j==k:
                    chiq_inv[j, k] = np.power(chi0q[i, j, j], -1)-coulomb[j, j]
                else:
                    chiq_inv[j, k] = np.power(chi0q[i, j, k], -1)
        chiq_m[i] = np.linalg.inv(chiq_inv)
    return chiq_m

def chi0z_XG(z, omega, n=0.025):
    qvec = zvec_to_qvec(z)
    chi0q = chi0q_XG(qvec, omega, n)
    nw = len(omega)
    nz = len(z)
    chi0z = np.zeros((nw, nz), dtype = complex)
    for i in range(nw):
        chi0z[i, :] = np.fft.ifft(DRF.Rev_vec(chi0q[i, :]))
    return chi0z

def chi0zz_XG(z, omega, n= 0.025):
    nw = len(omega)
    nz = len(z)
    chi0z = chi0z_XG(z, omega, n)
    chi0zz = np.zeros((nw, nz, nz), dtype = complex)
    chi0zz_out = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        chi0zz[i, 0, :] = chi0z[i, :]
    for i in range(nw):
        for j in range(1, nz):
            chi0zz[i, j, :] = np.append(chi0z[i, nz-j:nz], chi0z[i, 0:nz-j])
        chi0zz_out[i, :, :] = (chi0zz[i]+np.transpose(chi0zz[i]))/2
    return chi0zz_out

     
    


def chi0wzz_jellium(q, omega, n = 0.025):
    nw = len(omega)
    nz = len(q)
    chi0wz = np.zeros((nw, nz), dtype = complex)
    chi0wzz = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        chi0wz[i, :] = chi0_r_1d_test(q, omega[i], n = 0.025)
        chi0wzz[i, 0, :] = chi0wz[i, :]
    for i in range(nw):
        for j in range(1, nz):
            chi0wzz[i, j, :] = np.append(chi0wz[i, nz-j:nz], chi0wz[i, 0:nz-j])
    return chi0wzz

def chi0_q_1d_test(q, omega, n = 0.025):
    nq = len(q)
    chi0_q_1d = np.zeros((nq), dtype = complex)
    qrev = DRF.Rev_vec(q)
    qrev[0] = 0
    for i in range(nq):
        qi = qrev[i]**2
        norm_q = (qi)**(1/2)
        epsilon_q_1d = DRF.epsilon_1(norm_q, omega, n)+ic*DRF.epsilon_2(norm_q, omega, n)
        chi0_q_1d[i] = -(epsilon_q_1d-1)*e0*norm_q**2
    #kF = (3*math.pi**2*n)**(1/3)
    #if omega == 0:
    #    chi0_q_1d[0] = -4*kF/(4*math.pi**2)
    return chi0_q_1d

def chi0_r_1d_test(q, omega, n = 0.025):
    chi0_q = chi0_q_1d_test(q, omega)
    z = qvec_to_zvec(q)
    chi0z = np.fft.ifftn(chi0_q)
    size = chi0z.size
    chi0z = chi0z*size/max(z)
    return chi0z

def qvec_to_zvec(q):
    #Enter the q vector (-q, q) with the point q=0 so always an odd number of point
    npnt = len(q)
    center = math.floor(npnt/2)+1
    qmin = q[center]
    return np.linspace(0, 2*math.pi/qmin, npnt)
def zvec_to_qvec(z):
    #enter the z vector (0, zmax)
    npnt = len(z)
    zmin = z[1]
    qnosym = np.linspace(-math.pi/zmin, math.pi/zmin, npnt)
    qsym = np.zeros((npnt))
    for i in range(math.floor(npnt/2)+1):
        q = (-qnosym[i]+qnosym[npnt-i-1])/2
        qsym[i] = -q
        qsym[npnt-i-1] = q
    return qsym

def chi0_slab(thickness, dens, omega, q_p, d=75, n = 0.025, nband = 500, eta = 0.0036749326):
    if thickness%2==0:
        thickness+=1
    npoint = thickness*dens
    surf_lim = round(d*dens/2)
    z_b = np.linspace(0, thickness, dens*thickness)
    z_s = np.linspace(0, d, d*dens)
    qvec = zvec_to_qvec(z_b)
    chi0_bulk_wzz = chi0wzz_jellium(qvec, omega, n)
    chi0_slab_wzz = DRF.chi0wzz_slab_jellium_Eguiluz_1step_F(q_p, z_s, z_s, omega, n, d, eta)
    nw = len(omega)
    chi0wzz = chi0_bulk_wzz
    for i in range(nw):
        for j in range(surf_lim):
            for k in range(surf_lim):
                chi0wzz[i, j, k] = chi0_slab_wzz[i, j, k]
                chi0wzz[i, npoint-1-j, npoint-1-k] = chi0_slab_wzz[i, j, k]
    return chi0wzz

def Sym_chi_Slab(chi0wzz):
    nw, nz1, nz2 = chi0wzz.shape
    if nz1 == nz2:
        return chi0wzz
    else:
        chi0wzz_slab = np.zeros((nw, nz2, nz2), dtype = complex)
        for i in range(nw):
            chi0wzz_slab[i, 0:nz1, 0:nz2] = chi0wzz[i, :, :]
            for j in range(nz1):
                chi0wzz_slab[i, nz1-1+j, :] = chi0wzz[i, nz1-1-j, :][::-1]
        return chi0wzz_slab

def Despoja_2005(chi0wqq, qp, q_vec, d):
    nw, nz, nq = chi0wqq.shape
    eps_out = np.zeros((nw), dtype = complex)
    add = np.zeros((nq, nq))
    q2 = qp**2
    for i in range(nq):
        add[i, i] = q2+q_vec[i]**2
    add[0, 0] = add[0, 0]*2
    chi_out = np.zeros((nq, nq), dtype = complex)
    chi_inv = np.zeros((nq, nq), dtype = complex)
    for i in range(nw):
        chi_out = 8*math.pi/d*chi0wqq[i]+add
        #eps_out[i] = np.sum(chi_out[i])
        chi_inv = np.linalg.inv(chi_out)
        eps_out[i] = np.sum(chi_inv)
    eps_out = eps_out*(4*qp)/d
    return eps_out



def SF_Despoja_2005(eps, qp, d):
    eps1 = np.zeros((eps.shape), dtype = complex)
    for i in range(len(eps)):
        eps[i] = eps[i]**(-1)
        eps1[i] = eps[i]+1
    sf_out = np.imag(np.divide(eps, eps1))
    #*math.exp(-qp*d)*
    sf_out = 1/math.pi*math.exp(-qp*d)*2*sf_out
    return sf_out

def small_g(chiwqzz, z2, q_p):
    nw, nz1, nz2 = chiwqzz.shape
    g_func_int = np.zeros((nw, nz1, nz2), dtype = complex)
    exp_math = np.zeros((nz1, nz2))
    g_func_out = np.zeros(nw, dtype = complex)
    for i in range(nz1):
        for j in range(nz2):
            exp_math[i, j] = math.exp(q_p*(z2[i]+z2[j]))
    for i in range(nw):
        #,exp_math
        g_func_int[i] = chiwqzz[i]
        g_func_out[i] = np.sum(g_func_int[i])
    return -2*math.pi/q_p*g_func_out

def chiq_sorted(chiqwqq_in):
    nw, nq1, nq2 = chiqwqq_in.shape
    chiwq1q2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chiwq1q2[i, j, :] = Inv_Rev_vec_1(chiqwqq_in[i, j, :])
    chiwqq = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chiwqq[i, :, j] = Inv_Rev_vec_1(chiwq1q2[i, :, j])
    return chiwqq

def Fourier_dir(chi0wqq, q_vec):
    nw, nq1, nq2 = chi0wqq.shape
    q_vec = np.real(Inv_Rev_vec(q_vec))
    if nq1 !=nq2:
        raise ValueError("The matrix must have the same dimension nz1 and nz2")
    chi0wq1q2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chi0wq1q2[i, j, :] = Inv_Rev_vec(chi0wqq[i, j, :])
    chi0wqq = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chi0wqq[i, :, j] = Inv_Rev_vec(chi0wq1q2[i, :, j])
    chi0wzq2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq1):
            chi0wzq2[i, j, :] = np.fft.ifft(chi0wqq[i, j, :])
    chi0wz1z2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chi0wz1z2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j])
    q_sorted = Inv_Rev_vec(q_vec)
    chi0wz1z2 = chi0wz1z2*nq2**2/(2*math.pi)*q_sorted[1]
    chi0wz1z2_out = np.zeros((nw, nq1, nq2), dtype = complex)
    
    for i in range(nw):
        chi0wz1z2_out[i] = (chi0wz1z2[i, :, :]+np.transpose(chi0wz1z2[i, :, :]))/2
    return chi0wz1z2_out

def Inv_Rev_vec(Y):
    #In : [-3, -2, -1, 0, 0, 1, 2, 3] // [-3, -2, -1, 0, 1, 2, 3]
    #Out : [0, 1, 2, 3, -3, -2, -1, -0] // [0, 1, 2, 3, -3, -2, -1]
    l = Y.size
    Y_out = np.zeros((l), dtype = complex)
    if l%2==0:
        m = math.floor(l/2)
    else:
        m = math.floor(l/2)+1
    Y_out[0:m] = Y[m-1:l]
    Y_out[m:l] = Y[0:m-1]
    return Y_out

def Inv_Rev_vec_1(Y):
    #In : [-3, -2, -1, 0, 0, 1, 2, 3] // [-3, -2, -1, 0, 1, 2, 3]
    #Out : [0, 1, 2, 3, -3, -2, -1, -0] // [0, 1, 2, 3, -3, -2, -1]
    l = Y.size
    Y_out = np.zeros((l), dtype = complex)
    m = math.floor(l/2)
    Y_out[0:m] = Y[m+1:l]
    Y_out[m:l] = Y[0:m+1]
    return Y_out

def Fourier_dir_f(chi0wqq, q_vec):
    nw, nq1, nq2 = chi0wqq.shape
    chi0wzq2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq1):
            chi0wzq2[i, j, :] = np.fft.ifft(chi0wqq[i, j, :])
    chi0wz1z2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chi0wz1z2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j])
    chi0wz1z2 = chi0wz1z2*nq2**2/(2*math.pi)*q_vec[1]
    chi0wz1z2_out = np.zeros((nw, nq1, nq2), dtype = complex)    
    for i in range(nw):
        chi0wz1z2_out[i] = (chi0wz1z2[i, :, :]+np.transpose(chi0wz1z2[i, :, :]))/2
    return chi0wz1z2_out

def Fourier_dir_fwithout_sym(chi0wqq, q_vec):
    nw, nq1, nq2 = chi0wqq.shape
    chi0wzq2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq1):
            chi0wzq2[i, j, :] = np.fft.ifft(chi0wqq[i, j, :])
    chi0wz1z2 = np.zeros((nw, nq1, nq2), dtype = complex)
    for i in range(nw):
        for j in range(nq2):
            chi0wz1z2[i, :, j] = np.fft.ifft(chi0wzq2[i, :, j])
    chi0wz1z2 = chi0wz1z2*nq2
    return chi0wz1z2

def Fourier_inv_test(chi0wzz, z):
    nw, nz1, nz2 = chi0wzz.shape
    if nz1 !=nz2:
        raise ValueError("The matrix must have the same dimension nz1 and nz2")
    chi0wzq2 = np.zeros((nw, nz1, nz2), dtype = complex)
    for i in range(nw):
        for j in range(nz1):
            chi0wzq2[i, j, :] = np.fft.fft(chi0wzz[i, j, :])
    chi0wq1q2 = np.zeros((nw, nz1, nz2), dtype = complex)
    for i in range(nw):
        for j in range(nz2):
            chi0wq1q2[i, :, j] = np.fft.fft(chi0wzq2[i, :, j])
    return chi0wq1q2/nz2**2*max(z)

def Fourier_inv(chi0wzz, z):
    nw, nz1, nz2 = chi0wzz.shape
    if nz1 !=nz2:
        raise ValueError("The matrix must have the same dimension nz1 and nz2")
    chi0wzq2 = np.zeros((nw, nz1, nz2), dtype = complex)
    for i in range(nw):
        for j in range(nz1):
            chi0wzq2[i, j, :] = np.fft.fft(DRF.Rev_vec(chi0wzz[i, j, :]))
    chi0wq1q2 = np.zeros((nw, nz1, nz2), dtype = complex)
    for i in range(nw):
        for j in range(nz2):
            chi0wq1q2[i, :, j] = np.fft.fft(Inv_Rev_vec(chi0wzq2[i, :, j]))
    return chi0wq1q2/nz2**2*max(z)

def SP_model1(chi0zz_s, chi0z_b, n_bulk):
    print("There is no possible automatic check that the two meshs of the differents matrices match. This test must be carried out manually by the user")
    nw1, nz1, nz2 = chi0zz_s.shape
    nw2, nz = chi0z_b.shape
    if nw1 != nw2:
        raise ValueError("The frequencies for the matrix of the slab and for the bulk should be the same")
    nw = nw1
    if nz1 != round(nz2/2) and nz1 != round(nz2/2)+1:
        raise ValueError("The mesh for the two spatial variables of the slab matrix should be identical")
    for i in range(nw):
        chi0z_b[i, :] = DRF.Rev_vec(chi0z_b[i, :])
    chi0zz_s_rev = np.zeros((nw1, nz1, nz2), dtype = complex)
    for i in range(nw):
        for j in range(nz1):
            chi0zz_s_rev[i, j, :] = chi0zz_s[i, j, ::-1]
    for i in range(nw):
        for j in range(nz2):
            chi0zz_s_rev[i, :, j] = chi0zz_s_rev[i, ::-1, j]
    npnt = nz2+n_bulk*nz
    chi0_model1 = np.zeros((nw, npnt, npnt), dtype = complex)
    chi0_model1[:, 0:nz1, 0:nz2] = chi0zz_s 
    chi0_model1[:, npnt-nz1:npnt, npnt-nz2:npnt] = chi0zz_s_rev
    for i in range(nw):
        for j in range(nz1, npnt-nz1):
            if j < round(nz/2)+1:
                delta = round(nz/2)-j
                chi0_model1[i, j, 0:nz-delta] = chi0z_b[i, delta:nz]
            elif j >= npnt-(round(nz/2))-1:
                delta = -npnt+(round(nz/2)+1)+j
                chi0_model1[i, j, j-(round(nz/2)):npnt] = chi0z_b[i, 0:nz-delta]
            else:
                chi0_model1[i, j, j-(round(nz/2)):j+(round(nz/2)+1)] = chi0z_b[i, :]
    return chi0_model1

def SP1(L, s, omega, dens=1, n = 0.025, eta = 0.2):
    nw = len(omega)
    #Bulk
    nz = dens*L+1
    z_b = np.linspace(0, L, nz)
    qvec_b = zvec_to_qvec(z_b)
    chi0wzz_b = chi0zz_XG(z_b, omega, n)
    #Slab
    nz_s = s*dens+1
    z_s2 = np.linspace(0, s, nz_s)
    z_s1 = np.linspace(0, math.floor(s/2)+1, (math.floor(s/2)+1)*dens+1)
    chi0wzz_s = DRF.chi0wzz_slab_jellium_Eguiluz_1step_F(z_s1, z_s2, omega, n, s, eta)
    ##Add Symmetrisation
    #Second-Principle
    chi0wzz_SP = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        for j in range(nz):
            for k in range(nz):
                if (j>=nz_s or k>=nz_s) and (j<=nz-nz_s or k<=nz-nz_s):
                    chi0wzz_SP[i, j, k] = chi0wzz_b[i, j, k]
                elif j<=nz_s and k<=nz_s:
                    chi0wzz_SP[i, j, k] = chi0wzz_s[i, j, k]
                else:
                    chi0wzz_SP[i, j, k] = chi0wzz_s[i, nz-j, nz-k]
    return chi0wzz_SP


def SP1_debbug(chi0wzz_b, chi0wzz_s, zvec):
    nw, nz, nz_out = chi0wzz_b.shape
    nz_s = round(len(chi0wzz_s[0, 0, :])/2)
    #Second-Principle
    chi0wzz_SP = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        chi0zz_SP = np.zeros((nz, nz), dtype = complex)
        for j in range(nz):
            for k in range(nz):
                if (j>=nz_s or k>=nz_s) and (j<=nz-nz_s or k<=nz-nz_s):
                    chi0zz_SP[j, k] = chi0wzz_b[i, j, k]
                elif j<=nz_s and k<=nz_s:
                    chi0zz_SP[j, k] = chi0wzz_s[i, j, k]
                else:
                    chi0zz_SP[j, k] = chi0wzz_s[i, nz-j, nz-k]
        chi0wzz_SP[i] = (chi0zz_SP+np.transpose(chi0zz_SP))/2
    chi0wqq = Fourier_inv(chi0wzz_SP, zvec)
    qvec = zvec_to_qvec(zvec)
    chiwqq = chiq_mat(chi0wqq[:, round(nz/2):nz, round(nz/2):nz], qvec[round(nz/2):nz])
    return chiwqq


def DSF(chiq_mat, q, n):
    nw, nq = chiq_mat.shape
    inv_eps = np.zeros((nw, nq), dtype = complex)
    dsf = np.zeros((nw, nq))
    coulomb = np.zeros((nq))
    coulomb_inv = np.zeros((nq))
    coulomb[1:nq] = np.power(q[1:nq], -2)
    coulomb_inv[1:nq] = np.power(q[1:nq], 2)
    for i in range(nw):
        inv_eps[i, :] = [1]*nq+np.multiply(coulomb, chiq_mat[i, :], dtype = complex)
        dsf[i]= -1/n*np.multiply(coulomb_inv, np.imag(inv_eps[i]))
    return dsf

def dielectric_inv(chi, q):
    nw, nq = chi.shape
    dielectric = np.zeros((nw, nq), dtype = complex)
    q_sq_inv = np.zeros((nq))
    q_sq_inv[1:nq] = np.power(q[1:nq], -2)
    q_sq_inv[0] = 1/1e-5
    for i in range(nw):
        dielectric[i, :] = 1+4*math.pi*np.multiply(q_sq_inv, chi[i, :], dtype = complex)
    return dielectric


    
def Penzar1984(omega, q, q_p, n):
    re_chi0 = real_penzar(omega, q, q_p, n)
    im_chi0 = imag_penzar(omega, q, q_p, n)
    return re_chi0+ic*im_chi0

def real_penzar(omega, q, q_p, n):
    nw = len(omega)
    nq = len(q)
    q2 = q_p**2
    re_chi0_out = np.zeros((nw, nq, nq))
    E_F = (1/2)*(3*math.pi**2*n)**(2/3)
    prefac = (2*math.pi*q2)
    for i in range(nw):
        for j in range(nq):
            for k in range(nq):
                if j==k:
                    #print(re_chi0_XG(np.array([q[j]]), np.array([omega[k]]), n))
                    re_chi0_out[i, j, k] = np.real(re_chi0_XG(np.array([q[j]]), np.array([omega[k]]), n)[0, 0])*prefac
                else:
                    re_chi0_out[i, j, k] = U_penzar(q2, q[j], q[k], omega[i], E_F) + U_penzar(q2, q[j], -q[k], omega[i], E_F) +U_penzar(q2, q[j], q[k], -omega[i], E_F) +U_penzar(q2, q[j], -q[k], -omega[i], E_F)
    return re_chi0_out/prefac

def imag_penzar(omega, q, q_p, n):
    nw = len(omega)
    nq = len(q)
    q2 = q_p**2
    im_chi0_out = np.zeros((nw, nq, nq))
    E_F = (1/2)*(3*math.pi**2*n)**(2/3)
    prefac = -(2*math.pi*q2)
    for i in range(nw):
        for j in range(nq):
            for k in range(nq):
                if j==k:
                    im_chi0_out[i, j, k] = np.real(im_chi0_XG(np.array([q[j]]), np.array([omega[k]]), n)[0, 0])*prefac
                else:
                    im_chi0_out[i, j, k] = V_penzar(q2, q[j], q[k], omega[i], E_F) + V_penzar(q2, q[j], -q[k], omega[i], E_F) -V_penzar(q2, q[j], q[k], -omega[i], E_F) -V_penzar(q2, q[j], -q[k], -omega[i], E_F)
    return im_chi0_out/prefac
                

def U_penzar(qp2, q1, q2, omega, E_F):
    return (np.abs(qp2-q1*q2+2*omega)-np.real((qp2-q1*q2+2*omega)-4*qp2*cmath.sqrt(2*E_F-1/4*(q1+q2)**2)))*np.sign(qp2-q1*q2+2*omega)*Heaviside(2*E_F-1/4*(q1+q2))

def V_penzar(qp2, q1, q2, omega, E_F):
    return np.real((qp2-q1*q2+2*omega)-4*qp2*cmath.sqrt(2*E_F-1/4*(q1+q2)**2))

def Heaviside(a):
    if a>=0:
        return 1
    else:
        return 0
