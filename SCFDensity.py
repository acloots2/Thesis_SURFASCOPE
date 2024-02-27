import numpy as np
import numba
from numba import jit
import math
import cmath
import jellium_bulk as jb
import jellium_slab as js
import tools
import test_set as ts

import time
import plotly.graph_objects as go
import Second_P_models as SP
from scipy.optimize import curve_fit
import Sol_Schrod as ss
import potentials
import PF_Loos_UEG as ueg

def trial_density(bg_width, dens_bg, void, point_dens):
    dens_el = dens_bg*bg_width
    r_s = ueg.radius_from_density(dens_bg)
    d_wall = 30/8*(2*math.pi/3)**(2/3)*r_s
    z_slab = bg_width+2*d_wall
    npnt_slab = math.ceil(z_slab*point_dens)
    print(npnt_slab)
    if npnt_slab%2 == 0:
        npnt_slab+=1
    z_slab_vec = tools.center_z(np.linspace(0, z_slab, npnt_slab))
    dens_vec = np.zeros(npnt_slab)
    bg_dens_vec = np.zeros(npnt_slab)
    for i in range(npnt_slab):
        if z_slab_vec[i]<=-bg_width/2:
            dens_vec[i] = dens_el*(np.exp(1/d_wall*np.log(10**10)*(z_slab_vec[i]+bg_width/2)))
        elif -bg_width/2<z_slab_vec[i]<0:
            dens_vec[i] = dens_el*(np.exp(1/z_slab*np.log(0.9)*np.abs((z_slab_vec[i]+bg_width/2))))
            bg_dens_vec[i] = 1
        elif 0<=z_slab_vec[i]<bg_width/2:
            dens_vec[i] = dens_el*(np.exp(1/z_slab*np.log(0.9)*(np.abs(z_slab_vec[i]-bg_width/2))))
            bg_dens_vec[i] = 1
        else:
            dens_vec[i] = dens_el*(np.exp(-1/d_wall*np.log(10**10)*(z_slab_vec[i]-bg_width/2)))
    dens_vec = dens_vec*dens_el/np.sum(dens_vec)*point_dens
    bg_dens_vec = bg_dens_vec*dens_el/np.sum(bg_dens_vec)*point_dens
    npnt_void = round(void*point_dens)
    npnt_tot = 2*npnt_void+npnt_slab
    dens_vec_out = np.zeros(npnt_tot)
    bg_dens_vec_out = np.zeros(npnt_tot)
    dens_vec_out[npnt_void:-npnt_void] = dens_vec
    bg_dens_vec_out[npnt_void:-npnt_void] = bg_dens_vec
    z_out = np.linspace(0, z_slab+2*void, npnt_tot)
    return dens_vec_out, bg_dens_vec_out, z_out

def radius_jellium(dens, dimension = 3):
    if dimension == 3:
        return (3/(4*math.pi*dens))**(1/3)
    elif dimension == 2:
        return np.power(1/math.pi*np.power(dens, -1), 1/2)
    else: 
        return 1/(2*dens)  

def e_xc(dens, dimension = 3):
    npnt = len(dens)
    e_xc_vec = np.zeros((npnt), dtype = complex)
    for i in range(npnt):
        if dens[i] > 0:
            #rs = ueg.radius_from_density(dens[i], dimension)
            rs = radius_jellium(dens[i], dimension)
            e_xc_vec[i] = -0.458/rs+0.44/(rs+7.8)
    return e_xc_vec

def v_xc(e_xc_vec, dens):
    A = (-0.458*(4*math.pi/3)**(1/3))
    B = (0.44*(4*math.pi)**(1/3))
    C = 3**(1/3)
    D = (7.8*(4*math.pi)**(1/3))
    npnt = len(dens)
    v_xc_vec = np.zeros((npnt), dtype = complex)
    for i in range(npnt):
        dens_pow = dens[i]**(1/3)
        v_xc_vec[i] = e_xc_vec[i]+1/3*(A*dens_pow+B*C*np.divide(dens_pow,np.power(C+D*dens_pow, 2)))
    return v_xc_vec


def xc_pot(dens, sym = True):
    e_xc = np.multiply(dens,ueg.e_total(dens, 3))
    index = np.argwhere(e_xc)[:, 0]
    npnt = len(e_xc)
    v_xc = np.zeros(npnt)
    v_xc[index] = np.gradient(e_xc[index], dens[index])
    if sym:
        v_xc[math.ceil(npnt/2)-1] = v_xc[math.ceil(npnt/2)] 
    return v_xc



def electrostatic_pot(x, elec_dens, bg_dens, point_dens):
    npnt = len(elec_dens)
    elec_pot = np.zeros(npnt, dtype = complex)
    delta_dens = elec_dens-bg_dens
    dx = (x[1]-x[0])
    for i in range(npnt):
        integral = 0
        for j in range(npnt):
            integral += (np.abs(x[i]-x[j])*delta_dens[j])
        elec_pot[i] = integral
    elec_pot = (elec_pot+elec_pot[::-1])/2
    return -2*math.pi*elec_pot/point_dens

def density(bands, energies, e_f, point_dens = 1):
    index = 0
    npnt = len(energies)
    dens = np.zeros((npnt), dtype = complex)
    while energies[index] < e_f:
        dens += (e_f-energies[index])*bands[:, index]*np.conj(bands[:, index])
        index += 1
    return 1/math.pi*dens*point_dens

def sc_cycle(z_vec, dens0, bg_dens, d_slab, nmax0, dens_3d, point_dens, dimension = 3):
    fig_pot = go.Figure()
    fig_dens = go.Figure()
    title_pot = "Self-Consistent Potential"
    title_dens = "Self-Consistent Density"
    xtitle = r"$z$"
    ytitle_pot = r"$V(z)$"
    ytitle_dens = r"$\rho(z)$"
    """omega = np.array([0])
    eta =  0.05*0.03675
    qp = 0.01/1.8897259886"""
    dens_tot = dens_3d*d_slab
    #Computation of the electrostatic potential related to the input density
    pot0 = electrostatic_pot(z_vec, dens0, bg_dens, point_dens)
    # Computation of the xc_potential using PF Loos 2016
    e_xc_vec  = e_xc(dens0, dimension)
    vxc_0 = v_xc(e_xc_vec, dens0)*(d_slab)
    #vxc_0 = np.zeros(len(pot0))
    #vxc_0 = xc_pot(dens0)
    #total potential = es_pot+xc_pot
    print(np.sum(dens_tot))

    fig_pot.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(pot0), name = "es pot", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    """
    fig_pot.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(vxc_0), name = "xc pot", mode = "lines+markers",marker=dict(
    size=5,
    ),))"""
    pot0 += vxc_0
    pot0 = pot_eff(pot0, vxc_0, z_vec)/d_slab
    fig_pot.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(pot0), name = "es+xc start", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    fig_dens.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(dens0), name = "dens0", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    fig_dens.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(bg_dens), name = "bg dens", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    # Solution of the Schrodinger equation associated with the new potential
    energies_0, bands_0, e_f0, nmax0 = js.pre_run_chi0(pot0, z_vec, dens_3d, d_slab)
    print(nmax0)
    """#Computation of the precursor 
    chi0 = js.chi0wzz_slab_jellium_with_pot(qp, energies_0, bands_0, omega, e_f0, nmax0, max(z_vec), eta)
    diel_wov = js.sym_chi_slab(chi0)
    q_vec = tools.zvec_to_qvec(z_vec)
    diel_wov_q = js.fourier_inv(diel_wov, z_vec)
    eps = js.epsilon(diel_wov_q, np.real(tools.inv_rev_vec(q_vec)), qp)
    eps_inv = np.linalg.inv(eps)
    eps_inv = tools.fourier_dir(eps_inv)
    eps_inv = eps_inv[0]"""
    #Computation of the new densoty associated with this new system
    dens1 = density(bands_0, energies_0, e_f0, point_dens)

    #Mixing of the new density with the previous one
    
    #dens2 = dens0+np.matmul(eps_inv, dens1-dens0)
    dens2 = dens0+0.1*(dens1-dens0)

    #Symmetrisation
    dens2 = (dens2+dens2[::-1])/2
    dens2 = dens2*dens_tot/np.sum(dens2)*point_dens
    print(np.sum(dens2))

    fig_dens.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(dens1), name = "Cycle 0", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    fig_dens.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(dens2), name = "Cycle 1", mode = "lines+markers",marker=dict(
    size=5,
    ),))
    
    i = 0
    npnt = len(dens0)
    while np.sum(np.abs(dens0-dens2)/npnt)>1e-7 and i<10:
        print(np.sum(np.abs(dens0-dens2)/npnt))
        # The new density becomes the base density
        dens0 = dens2
        # The assoiated total potential is computed
        pot0 = electrostatic_pot(z_vec, dens0, bg_dens, point_dens)
        e_xc_vec  = e_xc(dens0, dimension)
        vxc_0 = v_xc(e_xc_vec, dens0)*d_slab
        #vxc_0 = xc_pot(dens0)
        #vxc_0 = np.zeros(len(pot0))
        pot0 += vxc_0
        pot0 = pot_eff(pot0, vxc_0, z_vec)/d_slab
        #The Schrodinger Equation is solved
        energies_0, bands_0, e_f0, nmax0 = js.pre_run_chi0(pot0, z_vec, dens_3d, d_slab)
        print(nmax0)
        #The new density and precursor are computed
        dens1 = density(bands_0, energies_0, e_f0, point_dens)
        """chi0 = js.chi0wzz_slab_jellium_with_pot(qp, energies_0, bands_0, omega, e_f0, nmax0, max(z_vec), eta)
        diel_wov = js.sym_chi_slab(chi0)
        q_vec = tools.zvec_to_qvec(z_vec)
        diel_wov_q = js.fourier_inv(diel_wov, z_vec)
        eps = js.epsilon(diel_wov_q, np.real(tools.inv_rev_vec(q_vec)), qp)
        eps_inv = np.linalg.inv(eps)
        eps_inv = tools.fourier_dir(eps_inv)
        eps_inv = eps_inv[0]"""
        # The two densities are mixed
        dens2 = dens0+0.1*(dens1-dens0)
        #dens2 = dens0+np.matmul(eps_inv, tools.rev_vec(dens1-dens0))
        dens2 = (dens2+dens2[::-1])/2
        dens2 = dens2*dens_tot/np.sum(dens2)*point_dens
        print(i)
        i+=1
        if i%2 ==0:
            fig_pot.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(pot0), name = "Cycle "+str(2+i), mode = "lines+markers",marker=dict(
        size=5,
        ),))
            fig_dens.add_trace(go.Scatter(x = tools.center_z(z_vec), y = np.real(dens2), name = "Cycle "+str(2+i), mode = "lines+markers",marker=dict(
        size=5,
        ),))
    fig_pot.update_layout(title_text = title_pot, title_x=0.5,xaxis_title= xtitle,
    yaxis_title = ytitle_pot, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',)
    fig_pot.show()
    fig_dens.update_layout(title_text = title_dens, title_x=0.5,xaxis_title= xtitle,
    yaxis_title = ytitle_dens, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',)
    fig_dens.show()
    return dens2, pot0


def pot_eff(pot_es, pot_xc, z_pot):
    x0, x1 = pot_solver(pot_es, pot_xc, z_pot)
    print(x0, x1)
    npnt = len(z_pot)
    pot_eff = np.zeros(npnt, dtype = complex)
    i = 0
    z_pot = tools.center_z(z_pot)
    if x1<0:
        while z_pot[i]<x1:
            pot_eff[i] = -1/4*(z_pot[i]+x0)
            i+=1
        while z_pot[i]<=-x1:
            pot_eff[i] = pot_es[i]+pot_xc[i]
            i+=1
        while i<npnt:
            pot_eff[i] = 1/4*(z_pot[i]-x0)
            i+=1
    else:
        while z_pot[i]<-x1:
            pot_eff[i] = 1/4*(z_pot[i]+x0)
            i+=1
        while z_pot[i]<=x1:
            pot_eff[i] = pot_es[i]+pot_xc[i]
            i+=1
        for j in range(i, npnt):
            pot_eff[i] = -1/4*(z_pot[i]-x0)
        
    return pot_eff

def pot_solver(pot_es, pot_xc, z_pot):
    dpot = np.gradient(pot_es+pot_xc, z_pot)
    i = 0
    slope = .1
    #print(dpot)
    if np.abs(dpot[i])>slope:
        while np.abs(dpot[i])>slope:
            i+=1
    else:
        while np.abs(dpot[i])<slope:
            i+=1
    z_fig = tools.center_z(z_pot)
    x1 = z_fig[i]
    pot_es1 = pot_es[i]
    pot_xc1 = pot_xc[i]
    x0 = 1/slope*(pot_es1+pot_xc1-slope*x1)
    return x0, x1
