"""Set of Set of Potentials for jellium"""

import math
import cmath
import numpy as np
import numba
from numba import jit
import Sol_Schrod as ss
import tools
import jellium_slab as js



ic = complex(0,1)
E0 = 1/(4*math.pi)

def square_well_pot(d_wall, void, well_height, point_dens):
    d_sys = d_wall+2*void
    npnt = round(d_sys*point_dens)
    if npnt%2 == 0:
        npnt+=1
    v_pot = np.zeros(npnt)
    nvoid = round(void*point_dens)
    v_pot[0:nvoid] = well_height
    v_pot[-nvoid::] = well_height
    z_pot = np.linspace(0, d_sys, npnt)
    return v_pot, z_pot

def square_well(d_wall, void, well_top, well_bottom, point_dens):
    d_sys = d_wall+2*void
    npnt = round(d_sys*point_dens)
    if npnt%2 == 0:
        npnt+=1
    v_pot = np.ones(npnt)*well_bottom
    nvoid = round(void*point_dens)
    v_pot[0:nvoid] = well_top
    v_pot[-nvoid::] = well_top
    z_pot = np.linspace(0, d_sys, npnt)
    return v_pot, z_pot

def system_optimizer(d_slab_start, dens_start, d_void, point_dens = 1):
    d_slab_inf = 1000
    #d_void = 250
    well_depth = 1000
    v_pot, z_pot = square_well_pot(d_slab_inf, d_void, well_depth, point_dens)
    energies, bands, e_f_inf, nmax = js.pre_run_chi0(v_pot, z_pot, dens_start, d_slab_inf)
    d_inf = np.real(3*math.pi/(8*cmath.sqrt(2*e_f_inf)))
    d_well = d_slab_start+2*d_inf
    rho = dens_start*d_slab_start/d_well
    v_pot, z_pot = square_well_pot(d_well, d_void, well_depth, point_dens)
    energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, rho, d_well)
    for i in range(5):
        d_well = np.real(d_wall(cmath.sqrt(2*e_f), d_slab_start))
        rho = dens_start*d_slab_start/d_well
        v_pot, z_pot = square_well_pot(d_well, d_void, well_depth, point_dens)
        energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, rho, d_well)
    print(d_well, rho, e_f)
    return v_pot, z_pot, energies, bands, e_f, nmax, rho, d_well

def d_wall(k_f, l):
    return l/2+1/(8*k_f)*(3*math.pi+cmath.sqrt(16*(k_f*l)**2+24*math.pi*k_f*l+25*math.pi**2))

def d_slab(k_f, d_w):
    return d_w - 2*(1/(8*k_f)*(3*math.pi+math.pi**2/(k_f*d_w)))

def potentiel_Schulte(d_slab, dens, d_void, well_top, well_bottom, point_dens):
    r_s = radius(dens)
    d_wall = 3/8*(2*math.pi/3)**2/3*r_s
    d_sys = d_slab+2*d_wall+2*d_void
    npnt = round(d_sys*point_dens)
    if npnt%2 == 0:
        npnt+=1
    v_pot = np.ones(npnt)*well_bottom
    nvoid = round(d_void*point_dens)
    v_pot[0:nvoid] = well_top
    v_pot[-nvoid::] = well_top
    z_pot = np.linspace(0, d_sys, npnt)
    return v_pot, z_pot

def radius(dens):
    return (3/(4*math.pi*dens))**(1/3)

def double_slab_pot(d_wall, d_void_ext, d_void_in, well_top, well_bottom, point_dens):
    d_sys = 2*d_wall+2*d_void_ext+d_void_in
    npnt = round(d_sys*point_dens)
    if npnt%2 == 0:
        npnt+=1
    double_pot = np.ones(npnt)*well_top
    npnt_void = round(d_void_ext*point_dens)
    npnt_slab = round(d_wall*point_dens)
    double_pot[npnt_void:npnt_void+npnt_slab] = well_bottom
    double_pot[-(npnt_void+npnt_slab):-npnt_void] = well_bottom
    z_pot = np.linspace(0, d_sys, npnt)
    return double_pot, z_pot

