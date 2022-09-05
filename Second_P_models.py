from asyncio import constants
from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
import time 
import abipy
from abipy.electrons.scr import ScrFile
import numpy as np
import cmath
import math
import pointcloud as pc
import DRF
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator

ic = complex(0,1)
e0 = 1/(4*math.pi)

def im_chi0_XG(q, omega, n=0.025):
    npnt = len(q)
    nw = len(omega)
    chi0q = np.zeros((nw, npnt))
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
            elif E_F>=E_minus and E_minus>= E_F-w:
                chi0q[i, j]=1/(2*math.pi)*(1/q_norm)*(E_F-E_minus)
            else:
                continue
    return -math.pi*chi0q

def re_chi0_XG(q, omega, n=0.025):
    npnt = len(q)
    nw = len(omega)
    chi0q = np.zeros((nw, npnt))
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
    for i in range(nw):
        chi0zz[i, 0, :] = chi0z[i, :]
    for i in range(nw):
        for j in range(1, nz):
            chi0zz[i, j, :] = np.append(chi0z[i, nz-j:nz], chi0z[i, 0:nz-j])
    return chi0zz

     
    


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

def chi0_slab(thickness, dens, omega, d=75, n = 0.025, nband = 500, eta = 0.0036749326):
    if thickness%2==0:
        thickness+=1
    npoint = thickness*dens
    surf_lim = round(d*dens/2)
    z_b = np.linspace(0, thickness, dens*thickness)
    z_s = np.linspace(0, d, d*dens)
    qvec = zvec_to_qvec(z_b)
    chi0_bulk_wzz = chi0wzz_jellium(qvec, omega, n)
    chi0_slab_wzz = DRF.chi0wzz_slab_jellium_Eguiluz_1step_F(z_s, z_s, omega, n, d, nband, eta)
    nw = len(omega)
    chi0wzz = chi0_bulk_wzz
    for i in range(nw):
        for j in range(surf_lim):
            for k in range(surf_lim):
                chi0wzz[i, j, k] = chi0_slab_wzz[i, j, k]
                chi0wzz[i, npoint-1-j, npoint-1-k] = chi0_slab_wzz[i, j, k]
    return chi0wzz
