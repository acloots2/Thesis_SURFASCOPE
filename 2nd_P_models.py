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


def chi0_slab(thickness, dens, omega, d=75, n = 0.025, nband = 500, eta = 0.0036749326):
    npoint = thickness*dens
    surf_lim = np.round(d*dens/2)
    z_b = np.linspace(0, thickness, dens*thickness)
    z_s = np.linspace(0, d, d*dens)
    qvec = DRF.zvec_to_qvec(z_b)
    chi0_bulk_wzz = DRF.chi0wzz_jellium(max(qvec), dens, omega, n)
    chi0_slab_wzz = DRF.chi0wzz_slab_jellium_Eguiluz_1step_F(z_s, z_s, omega, n, d, nband, eta)
    nw = len(omega)
    chi0wzz = chi0_bulk_wzz
    for i in range(nw):
        for j in range(surf_lim):
            for k in range(surf_lim):
                chi0wzz[i, j, k] = chi0_slab_wzz[i, j, k]
    for i in range(nw):
        for j in range(surf_lim):
            for k in range(surf_lim):
                chi0wzz[i, npoint-1-j, npoint-1-k] = chi0_slab_wzz[i, j, k]
    return chi0wzz
