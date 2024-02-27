""" This file contains a synthesis of all the work done about jellium slabs.
    A basic build is provided: describe some space, the background charges and the potential
    then you can derive all the different energies and dielectric properties.

    Different visualization tools are provided in order to ease the understanding of the data.

    A self-consistent scheme (for the relation between the potential and the density to be fullfiled) is on its way.
    At the moment, the code assumes an LDA exchange and correlation energy
 """

import numpy as np
import math
import tools
import potentials
import plotly.graph_objects as go
import jellium_slab as js
import test_set as ts






def radius_jellium(dens_elec, dimension = 3):
    if dimension == 3:
        return (3/(4*math.pi*dens_elec))**(1/3)
    elif dimension == 2:
        return np.power(1/math.pi*np.power(dens_elec, -1), 1/2)
    else: 
        return 1/(2*dens_elec) 
    
def background_dens(v_pot, dens_sys, d):
    if np.min(v_pot) == 0:
        index = np.argwhere(v_pot-np.max(v_pot))
    else:
        index = np.argwhere(v_pot)
    nz = len(v_pot)
    bg = np.zeros(nz)
    bg_ones = np.ones(nz)
    bg[index] = bg_ones[index]
    dens_bg = bg*d*dens_sys/np.sum(bg)
    return dens_bg

def density(energies, bands, e_f):
    index = 0
    npnt = len(energies)
    dens = np.zeros((npnt))
    while energies[index] < e_f:
        bands_dot = bands[:, index]*np.conj(bands[:, index])
        if np.max(np.imag(bands_dot))>1e-20:
            raise ValueError("It seems the eigen function are not orthonormal")
        dens += (e_f-energies[index])*np.real(bands_dot)
        index += 1
    return 1/math.pi*dens

def kinetic_energy_test(bands, nband_occ, z_vec):
    e_kin = 0
    for i in range(nband_occ):
        band_grad = np.gradient(bands[:, i], z_vec)
        band_grad = np.gradient(band_grad, z_vec)
        e_kin += -np.sum(np.multiply(np.conj(bands[:, i]), band_grad))
    return e_kin/2*np.abs(z_vec[0]-z_vec[1])

def kinetic_energy(bands, nband_occ, z_vec):
    e_kin = 0
    for i in range(nband_occ):
        band_grad = np.gradient(bands[:, i], z_vec)
        e_kin += np.sum(np.multiply(np.conj(band_grad), band_grad))
    return e_kin/2*np.abs(z_vec[0]-z_vec[1])

#xc energy/electron
def e_xc(dens_elec, dimension = 3):
    npnt = len(dens_elec)
    e_xc_vec = np.zeros((npnt), dtype = complex)
    for i in range(npnt):
        if dens_elec[i] > 0:
            #rs = ueg.radius_from_density(dens[i], dimension)
            rs = radius_jellium(dens_elec[i], dimension)
            e_xc_vec[i] = -0.458/rs+0.44/(rs+7.8)
    return e_xc_vec

#xc energy
def xc_Energy(dens_elec, z_vec):
    exc = e_xc(dens_elec)
    Exc = np.dot(exc, dens_elec)
    return Exc*np.abs(z_vec[0]-z_vec[1])

def xc_energy(vxc_pot, dens_elec, delta):
    return np.sum(np.multiply(vxc_pot, dens_elec))*delta


#xc potential
def xc_pot(dens_elec):
    dens_inv = np.power(dens_elec, -1/3)
    A = -(0.0595536*np.power(dens_inv, 2)+24.3776*dens_inv+239.561)
    B = np.power((1.2407*dens_inv+15.6), 2)*dens_inv
    return np.divide(A, B)

def pot_energy(xc_vec, hartree_vec, dens_elec, delta):
    tot_pot = xc_vec+hartree_vec
    return np.sum(np.multiply(tot_pot, dens_elec))*delta



def solve_pot(pot, z):
    dz = np.abs(z[1]-z[0])
    pot_prime = (pot[2]-pot[1])/dz
    pot_sec = (pot[2]-2*pot[1]+pot[0])/dz**2
    x1 = z[2]
    x2 = z[4]
    y1 = pot[2]
    y2 = pot[4]
    A = np.array([[x1**3, x1**2, x1, 1], [3*x1**2, 2*x1, 1, 0], [6*x1, 2, 0, 0], [x2**3, x2**2, x2, 1]])
    b = np.array([y1, pot_prime, pot_sec, y2])
    return np.linalg.solve(A, b)

#hartree energy
def hartree_energy_t(elec_dens, bg_dens, z_vec):
    npnt = len(elec_dens)
    delta_dens = elec_dens-bg_dens
    dz = np.zeros((npnt, npnt))
    #delta_dens = np.matmul(np.transpose(np.array([delta_dens])), np.array([delta_dens]))
    for i in range(npnt):
        for j in range(npnt):
            dz[i, j] = np.abs(z_vec[i]-z_vec[j])
    return -math.pi*np.matmul(np.matmul(dz, delta_dens)*np.abs(z_vec[0]-z_vec[1]), np.transpose(delta_dens))*np.abs(z_vec[0]-z_vec[1])

def hartree_energy(vh_pot, dens_elec, delta):
    return np.sum(np.multiply(vh_pot, dens_elec))*delta

#
def electrostatic_pot(x, elec_dens, bg_dens):
    npnt = len(elec_dens)
    elec_pot = np.zeros(npnt, dtype = complex)
    delta_dens = elec_dens-bg_dens
    dz = np.zeros((npnt, npnt))
    for i in range(npnt):
        for j in range(npnt):
            dz[i, j] = np.abs(x[i]-x[j])
    elec_pot = np.matmul(dz, delta_dens)
    elec_pot = (elec_pot+elec_pot[::-1])/2
    return -2*math.pi*elec_pot

def tot_energy(elec_dens, bg_dens, bands, nband_occ, z_vec):
    kin = kinetic_energy(bands, nband_occ, z_vec)
    hartree = hartree_energy(elec_dens, bg_dens, z_vec)
    xc = xc_Energy(elec_dens, z_vec)
    return kin+hartree+xc




def coulomb2d(q_paral, z_pot):
    nz = len(z_pot)
    delta = np.zeros((nz, nz))
    z_pot = tools.center_z(z_pot)
    for i in range(nz):
        for j in range(nz):
            delta[i, j] = np.abs(z_pot[i]-z_pot[j])
    return 2*math.pi/q_paral*np.exp(-q_paral*delta)

def eps_inv(eps_2d):
    nw, nz = eps_2d.shape[0], eps_2d.shape[1]
    eps_2d_i = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        eps_2d_i[i] = np.linalg.inv(eps_2d[i])
    return eps_2d_i

def dielectric_2d(chi0, coulomb_2d, dz):
    nw, nz = chi0.shape[0], chi0.shape[1]
    diel2d = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        diel2d[i] = np.diag(np.ones(nz))/dz-np.matmul(coulomb_2d, chi0[i])*dz
    return diel2d

def dielectric_inv_2d(chi_2d, coulomb_2d, dz):
    nw, nz = chi_2d.shape[0], chi_2d.shape[1]
    diel_i2d = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        diel_i2d[i] = np.diag(np.ones(nz))/dz+np.matmul(coulomb_2d, chi_2d[i])*dz
    return diel_i2d

def chi_from_epsilon(eps_2d, coulomb_2d, dz):
    nw, nz = eps_2d.shape[0], eps_2d.shape[1]
    eps_inv_2d = eps_inv(eps_2d)
    chi_2d = np.zeros((nw, nz, nz), dtype = complex)
    coulomb_inv = np.linalg.inv(coulomb_2d)
    for i in range(nw):
        chi_2d[i] = np.matmul(coulomb_inv, eps_inv_2d[i]-np.diag(np.ones(nz))/dz)*dz
    return chi_2d


def chi_from_epsilon_inv(eps_inv_2d, coulomb_2d, dz):
    nw, nz = eps_inv_2d.shape[0], eps_inv_2d.shape[1]
    chi_2d = np.zeros((nw, nz, nz), dtype = complex)
    coulomb_inv = np.linalg.inv(coulomb_2d)
    for i in range(nw):
        chi_2d[i] = np.matmul(coulomb_inv, eps_inv_2d[i]-np.diag(np.ones(nz))/dz)*dz
    return chi_2d

def screened_int_from_chi(chi_2d, coulomb_2d, dz):
    nw, nz = chi_2d.shape[0], chi_2d.shape[1]
    w_2d = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        w_2d[i] = coulomb_2d + np.matmul(np.matmul(coulomb_2d, chi_2d[i]), coulomb_2d)*dz**2
    return w_2d

def scf_fermi_level(d, d_void, top, bottom, density, ext_en, pnt_dens):
    v_pot, z_pot = potentials.square_well(d, d_void, top, bottom, pnt_dens)
    energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, density, d)
    e_f0=0
    while np.abs(np.real(e_f-e_f0))>1e-8:
        e_f0 =e_f
        top = np.real(e_f)+ext_en
        v_pot, z_pot = potentials.square_well(d, d_void, top, bottom, pnt_dens)
        energies, bands, e_f, nmax = js.pre_run_chi0(v_pot, z_pot, density, d)
        print(e_f)
    return energies, bands, e_f, nmax, v_pot, z_pot

def find_well_width(z_vec, pot):
    if np.min(pot) == 0:
        index = np.argwhere(pot-np.max(pot))
    else:
        index = np.argwhere(pot)
    z_out = z_vec[index]
    return np.max(z_out)-np.min(z_out), len(index)

def scf_cycle_energy(v_pot, z_pot, dens, dens_i, d_sys, qp = 0.05, alpha = 1e-7):
    
    #Figures
    fig_energy = go.Figure()
    fig_dens = go.Figure()
    fig_pot = go.Figure()
    
    #Starting point:
    system_int = jellium_slab(z_pot, v_pot, dens, qp = qp, d = d_sys)
    system_int.add_densities()
    dens_0 = system_int.dens_elec
    
    #Energy analysis arrays
    e_tot_vec = np.array([])
    e_kin_vec = np.array([])
    e_hartree_vec = np.array([])
    e_xc_vec = np.array([])
    e_pot_vec = np.array([])
    
    #Figure Initial density and pot
    fig_dens.add_trace(go.Scatter(x = tools.center_z(system_int.z), y = np.real(dens_0), name = "Starting e-dens", mode = "lines+markers", marker=dict(
    size=5,
    ),))
    fig_dens.add_trace(go.Scatter(x = tools.center_z(system_int.z), y = np.real(dens_i), name = "Bg_dens", mode = "lines+markers", marker=dict(
    size=5,
    ),))
    fig_pot.add_trace(go.Scatter(x = tools.center_z(system_int.z), y = np.real(v_pot), name = "Starting Pot", mode = "lines+markers", marker=dict(
    size=5,
    ),))
    
    #Compute energies of the sarting point
    system_int.add_energies()
    #e_tot_vec = np.append(e_tot_vec, system_int.tot_energy)
    e_tot0 = system_int.tot_energy
    e_kin_vec= np.append(e_kin_vec, system_int.kin_energy)
    
    
    
    #Compute the potential associated to the starting point
    pot_el = electrostatic_pot(z_pot, dens_0, dens_i)
    pot_xc = xc_pot(dens_0)
    e_xc_vec= np.append(e_xc_vec, xc_energy(pot_xc, system_int.dens_elec, system_int.delta_z))
    e_hartree_vec = np.append(e_hartree_vec, hartree_energy(pot_el, system_int.dens_elec, system_int.delta_z))
    e_pot_vec = np.append(e_pot_vec, pot_energy(pot_xc, pot_el, system_int.dens_elec, system_int.delta_z))
    #Compute the preconditionner
    system_int.add_df_inv()
    
    #Compute the new potential
    v_pot_1 = v_pot+np.matmul(system_int.df_inv, pot_el+pot_xc-v_pot)*system_int.delta_z
    v_pot_1 = v_pot_1[0, :]
    v_pot_1 = (v_pot_1+v_pot_1[::-1])/2
    
    #Create a new system with the new potential
    system_int = jellium_slab(z_pot, v_pot_1, dens, qp = qp, d = d_sys)
    system_int.add_densities()
    dens_1 = system_int.dens_elec
    #Density and potential after first cycle
    fig_dens.add_trace(go.Scatter(x = tools.center_z(z_pot), y = np.real(dens_1), name = "Dens 1", mode = "lines+markers", marker=dict(
    size=5,
    ),))
    fig_pot.add_trace(go.Scatter(x = tools.center_z(z_pot), y = np.real(v_pot_1), name = "Pot 1", mode = "lines+markers", marker=dict(
    size=5,
    ),))
    
    #Compute the energies of the new system
    system_int.add_energies()
    
    e_tot_vec = np.append(e_tot_vec, system_int.tot_energy)
    e_tot1 = system_int.tot_energy
    #e_kin_vec= np.append(e_kin_vec, system_int.kin_energy)
    
    #Begining of the scf cycle
    i = 0
    tol = alpha*max(system_int.z)
    while np.sum(np.abs(dens_0-dens_1))*system_int.delta_z>tol and i<25:
        e_tot0 = e_tot1
        dens_0 = dens_1
        #New Pot
        pot_el = electrostatic_pot(z_pot, system_int.dens_elec, dens_i)
        pot_xc = xc_pot(system_int.dens_elec)
        e_pot_vec = np.append(e_pot_vec, pot_energy(pot_xc, pot_el, system_int.dens_elec, system_int.delta_z))
        e_xc_vec= np.append(e_xc_vec, xc_energy(pot_xc, system_int.dens_elec, system_int.delta_z))
        e_hartree_vec = np.append(e_hartree_vec, hartree_energy(pot_el, system_int.dens_elec, system_int.delta_z))
        system_int.add_df_inv()
        v_pot_1 = v_pot_1+np.matmul(system_int.df_inv, pot_el+pot_xc-v_pot_1)*system_int.delta_z
        v_pot_1 = v_pot_1[0, :]
        v_pot_1 = (v_pot_1+v_pot_1[::-1])/2
        #Create a new system with the new potential
        system_int = jellium_slab(z_pot, v_pot_1, dens, qp = qp, d = d_sys)
        system_int.add_densities()
        dens_1 = system_int.dens_elec
        #Add each step to the figures
        fig_dens.add_trace(go.Scatter(x = tools.center_z(z_pot), y = np.real(dens_1), name = "Dens "+str(i+2), mode = "lines+markers", marker=dict(
        size=5,
        ),))
        fig_pot.add_trace(go.Scatter(x = tools.center_z(z_pot), y = np.real(pot_el), name = "Pot "+str(i+2), mode = "lines+markers", marker=dict(
        size=5,
        ),))
        
        #Compute the energies of the new system
        system_int.add_energies()
        #e_tot_vec = np.append(e_tot_vec, system_int.tot_energy)
        e_tot1 = system_int.tot_energy
        e_kin_vec= np.append(e_kin_vec, system_int.kin_energy)
        #e_hartree_vec = np.append(e_hartree_vec, system_int.hartree_energy)
        #e_xc_vec= np.append(e_xc_vec, -system_int.xc_energy)
        i += 1
    
    fig_energy.add_trace(go.Scatter(y = np.real(e_kin_vec)+np.real(e_hartree_vec)+np.real(e_xc_vec)*system_int.nmax, name = "E_tot", mode = "lines+markers", marker=dict(
        size=5,
        ),))
    fig_energy.add_trace(go.Scatter(y = np.real(e_kin_vec), name = "E_kin", mode = "lines+markers", marker=dict(
        size=5,
        ),))
    fig_energy.add_trace(go.Scatter(y = np.real(e_hartree_vec), name = "E_ha", mode = "lines+markers", marker=dict(
        size=5,
        ),))
    fig_energy.add_trace(go.Scatter(y = np.real(e_xc_vec), name = "E_xc", mode = "lines+markers", marker=dict(
        size=5,
        ),))
    fig_energy.add_trace(go.Scatter(y = np.real(e_pot_vec), name = "E_xc", mode = "lines+markers", marker=dict(
        size=5,
        ),))
    fig_energy.update_layout(title_text = r'SCF Energies', title_x=0.5,xaxis_title= r"$z$",
        yaxis_title = r'$E \text{ [Ha]}$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_dens.update_layout(title_text = r'SCF Densities', title_x=0.5,xaxis_title= r"$z$",
        yaxis_title = r'$\rho(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    fig_pot.update_layout(title_text = r'SCF potentials', title_x=0.5,xaxis_title= r"$z$",
        yaxis_title = r'$V(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    
    fig_energy.show()
    fig_dens.show()
    fig_pot.show()
    if i == 25:
        raise ValueError("The scf scheme has not found a converged solution after 25 iterations")
    return v_pot_1

def show_1d_quantity(self, quantity, title = "Title", xaxis = "x", yaxis = "y"):
        fig = go.Figure()
        xtitle = xaxis

        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(quantity), name = r"Real part", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.imag(quantity), name = r"Imaginary part", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

def show_1d(self, prop, z, d, w, i, title, name_1, name_2, yaxis):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        y1 = np.real(prop[w, i, :])
        y2 = np.imag(prop[w, i, :])
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y1, name = name_1, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y2, name = name_2, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        
        fig.add_annotation(
            x=tools.center_z(z)[i],
            y=np.real(prop[w, i, i]),
            text=r'$z_1$',
            showarrow=True,
            xanchor="right",
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            ax=50,
            ay=-30,
            font=dict(
            size=16,
            ),
        )
        fig.add_trace(go.Scatter(x = np.array([d/2, d/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        fig.add_trace(go.Scatter(x = np.array([-d/2, -d/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        fig.show()

def show_1d_omega(prop, omega, z, i, title, name_1, name_2, yaxis):
        fig = go.Figure()
        xtitle = r"$\omega \text{ eV}$"
        prop_diag = tools.extract_diag(prop)
        fig.add_trace(go.Scatter(x = omega*27.211, y = np.real(prop_diag[:, i]), name = name_1, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = omega*27.211, y = np.imag(prop_diag[:, i]), name = name_2, mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_annotation(
            xref="paper", 
            yref="paper",
            x=0.8, 
            y=0.9,
            text="z = "+str(tools.center_z(z)[i])+" Bohr",
            showarrow=False,
            font=dict(color = "black",
            size=13,
            ),
        )
        
        fig.update_layout(title_text = title, title_x=0.5,xaxis_title= xtitle,
                yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=True)
        fig.show()

def show_map(prop, z, omega, width, nz, nz_slab, w, title):
    fig = go.Figure(data =
    go.Contour(x = tools.center_z(z), y = tools.center_z(z),
        z=prop[w], name = "Freq = "+str(omega[w]*27.211)+" eV"
    ))
    fig.update_layout(title_text = title, title_x=0.5,xaxis_title= r'$z_1$',
            yaxis_title = r'$z_2$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = True)
    fig.add_trace(go.Scatter(x = np.ones(nz_slab)*width/2, y = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False))
    fig.add_trace(go.Scatter(x = -np.ones(nz_slab)*width/2, y = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_trace(go.Scatter(y = np.ones(nz_slab)*width/2, x = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_trace(go.Scatter(y = -np.ones(nz_slab)*width/2, x = np.linspace(-width/2, width/2, nz_slab), mode = "lines", line=dict(color = "black", dash =  'dot'),showlegend = False,))
    fig.add_annotation(
        x=z[round(nz*0.25)],
        y=z[round(nz*0.25)],
        text="Freq = "+str(omega[w]*27.211)+" eV",
        showarrow=False,
        font=dict(color = "black",
        size=16,
        ),
    )
    fig.show()

def show_map_omega(prop, z, omega, width, nz, title):
    ytitle = r"$\omega \text{ [eV]}$"
    prop_diag = np.real(tools.extract_diag(prop))
    fig = go.Figure(data =
    go.Contour(x = tools.center_z(z), y = omega*27.211,
        z=prop_diag,
    ))
    fig.update_layout(title_text = title, title_x=0.5,xaxis_title= r'$z$',
            yaxis_title = ytitle,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend = False)
    fig.update_traces(colorscale="Deep", selector=dict(type='contour'))
    fig.add_trace(go.Scatter(x = np.ones(nz)*width/2, y = omega*27.211, mode = "lines", line=dict(color = "black", dash =  'dash'),))
    fig.add_trace(go.Scatter(x = -np.ones(nz)*width/2, y = omega*27.211, mode = "lines", line=dict(color = "black", dash =  'dash'),))
    fig.show()




class jellium_slab:
    #This object only works for symmetrical slabs!!
    def __init__(self, z_vec, pot, dens, qp = 0, omega = np.array([0]), eta = 0.1*0.03675, d = [], dens_bg = []):
        self.z = z_vec
        self.dens_3d = dens
        self.tot_pot = pot
        self.nz = len(self.z)
        self.delta_z = np.abs(self.z[0]-self.z[1])
        if d == []:
            self.width, self.nz_slab = find_well_width(self.z, self.tot_pot)
        else:
            self.width, self.nz_slab = d, round(d/self.delta_z)
        self.dens_2d = self.width*self.dens_3d
       
        self.qp = qp
        self.omega = omega
        self.damping = eta
        self.nmax = 0
        self.plasma_freq = math.sqrt(self.dens_3d*4*math.pi) 
        self.dens_bg = dens_bg

        self.drf0 = []
        self.coulomb = []
        self.df = []
        self.df_inv = []
        self.drf = []
        self.scr_int = []
        self.loss = []
        
    def add_shrodinger_sol(self):
        self.energies, self.bands, self.e_f, self.nmax = js.pre_run_chi0(self.tot_pot, self.z, self.dens_3d, self.width)

    def add_densities(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.dens_elec = density(self.energies, self.bands, self.e_f)
        if self.dens_bg == []:
            z = tools.center_z(self.z)
            dens_b = np.zeros(self.nz)
            for i in range(self.nz):
                if np.abs(z[i])<self.width/2:
                    dens_b[i] = 1
            self.dens_bg = dens_b*(np.sum(self.dens_elec)*self.delta_z)/(np.sum(dens_b)*self.delta_z)

    def add_energies(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.kin_energy = kinetic_energy(self.bands, self.nmax, self.z)
        self.hartree_energy = hartree_energy(self.dens_elec, self.dens_bg, self.z)
        self.xc_energy = xc_Energy(self.dens_elec, self.z)
        self.tot_energy = self.kin_energy+self.hartree_energy+self.xc_energy
    
    def update_omega(self, omega):
        self.omega = omega
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()
        

    def update_qp(self, qp):
        self.qp = qp
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()

    def update_damping(self, damping):
        self.omega = damping
        if self.scr_int != []:
            self.update_diel_properties()
        elif self.drf != []:
            self.update_drf()
        elif self.df_inv != []:
            self.update_df_inv()
        elif self.df != []:
            self.update_df()
        elif self.drf0 != []:
            self.add_drf0()
        if self.loss != []:
            self.add_loss()


    def add_drf0(self):
        if self.nmax == 0:
            self.add_shrodinger_sol()
        self.drf0 = js.chi0wzz_slab_jellium_with_pot(self.qp, self.energies, self.bands, self.omega, self.e_f, self.nmax, max(self.z), self.damping)
        self.drf0 = js.sym_chi_slab(self.drf0)

    def add_diel_properties(self):
        self.coulomb = coulomb2d(self.qp, self.z)
        if self.drf0 == []:
            self.add_drf0()
        self.df = dielectric_2d(self.drf0, self.coulomb, self.delta_z)
        self.df_inv = eps_inv(self.df)
        self.drf = chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
        self.scr_int = screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def add_coulomb(self):
        self.coulomb = coulomb2d(self.qp, self.z)
    
    def add_df(self):
        if self.drf0 == []:
            self.add_drf0()
        if self.coulomb == []:
            self.add_coulomb()
        self.df = dielectric_2d(self.drf0, self.coulomb, self.delta_z)

    def add_df_inv(self):
        if self.df == []:
            self.add_df()
        self.df_inv = eps_inv(self.df)
    
    def add_drf(self):
        if self.df_inv == []:
            self.add_df_inv()
        self.drf = chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
    
    def add_scr_int(self):
        if self.drf == []:
            self.add_drf()
        self.scr_int = screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def update_diel_properties(self):
        self.coulomb = coulomb2d(self.qp, self.z)
        self.add_drf0()
        self.df = dielectric_2d(self.drf0, self.coulomb, self.delta_z)
        self.df_inv = eps_inv(self.df)
        self.drf = chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
        self.scr_int = screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def update_coulomb(self):
        self.coulomb = coulomb2d(self.qp, self.z)
    
    def update_df(self):
        self.add_drf0()
        self.coulomb = self.update_coulomb()
        self.df = dielectric_2d(self.drf0, self.coulomb, self.delta_z)

    def update_df_inv(self):
        self.df = self.update_df()
        self.df_inv = eps_inv(self.df)
    
    def update_drf(self):
        self.df_inv = self.update_df_inv()
        self.drf = chi_from_epsilon_inv(self.df_inv, self.coulomb, self.delta_z)
    
    def update_scr_int(self):
        self.drf = self.update_drf()
        self.scr_int = screened_int_from_chi(self.drf, self.coulomb, self.delta_z)

    def add_loss(self):
        self.loss, self.weights, self.eig_q, self.eps, self.vec, self.vec_dual = ts.loss_full_slab_wov(self.drf0, self.z, self.qp) 

    def show_pot(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.tot_pot), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Potential of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$V_{tot}$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_density_elec(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_elec), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Electronic density of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_density_bg(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_bg), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Background density of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho^+(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_densities(self):
        fig = go.Figure()
        xtitle = r"$z\text{ [Bohr]}$"

        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_elec), name = r"$\rho^-$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = np.real(self.dens_bg), name = r"$\rho^+$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Densities of the system', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$\rho(z)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_1d_drf0(self, w, i):
        show_1d(self, self.drf0, self.z, self.width, w, i, r'$\chi^0\text{ for a perturbation in }z_1$', r'$\Re{\chi^0}$', r'$\Im{\chi^0}$', r'$\chi^0(\omega, z_1, z)$')
        
    def show_1d_drf(self, w, i):
        show_1d(self, self.drf, self.z, self.width, w, i, r'$\chi\text{ for a perturbation in }z_1$', r'$\Re{\chi}$', r'$\Im{\chi}$', r'$\chi(\omega, z_1, z)$')
    
    def show_1d_df(self, w, i):
        show_1d(self, self.df, self.z, self.width, w, i, r'$\epsilon\text{ for a perturbation in }z_1$', r'$\Re{\epsilon}$', r'$\Im{\epsilon}$', r'$\epsilon(\omega, z_1, z)$')

    def show_1d_df_inv(self, w, i):
        show_1d(self, self.df_inv, self.z, self.width, w, i, r'$\epsilon^{-1}\text{ for a perturbation in }z_1$', r'$\Re{\epsilon^{-1}}$', r'$\Im{\epsilon^{-1}}$', r'$\epsilon^{-1}(\omega, z_1, z)$')

    def show_1d_scr_int(self, w, i):
        show_1d(self, self.df_inv, self.z, self.width, w, i, r'$W\text{ for a perturbation in }z_1$', r'$\Re{W}$', r'$\Im{W}$', r'$W(\omega, z_1, z)$')


    def show_1d_drf0_omega(self, i):
        show_1d_omega(self.drf0, self.omega, self.z, i, r'$\chi^0 \text{ for a perturbation and response in z}$', r'$\Re{\chi^0}$', r'$\Im{\chi^0}$', r'$\chi^0(\omega, z, z)$')


    def show_1d_drf_omega(self, i):
        show_1d_omega(self.drf, self.omega, self.z, i, r'$\chi \text{ for a perturbation and response in z}$', r'$\Re{\chi}$', r'$\Im{\chi}$', r'$\chi(\omega, z, z)$')

    
    def show_1d_df_omega(self, i):
        show_1d_omega(self.df, self.omega, self.z, i, r'$\epsilon \text{ for a perturbation and response in z}$', r'$\Re{\epsilon}$', r'$\Im{\epsilon}$', r'$\epsilon(\omega, z, z)$')

    def show_1d_df_inv_omega(self, i):
        show_1d_omega(self.df_inv, self.omega, self.z, i, r'$\epsilon^{-1} \text{ for a perturbation and response in z}$', r'$\Re{\epsilon^{-1}}$', r'$\Im{\epsilon^{-1}}$', r'$\epsilon^{-1}(\omega, z, z)$')

    def show_1d_scr_int_omega(self, i):
        show_1d_omega(self.scr_int, self.omega, self.z, i, r'$W \text{ for a perturbation and response in z}$', r'$\Re{W}$', r'$\Im{W}$', r'$W(\omega, z, z)$')


    def show_map_drf0(self, w, compl = False):
        if compl:
            show_map(np.imag(self.drf0), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\chi^0}(z_1, z_2)$')
        else:
            show_map(np.real(self.drf0), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\chi^0}(z_1, z_2)$')

    def show_map_drf0_omega(self, compl = False):
        if compl:
            show_map_omega(np.imag(self.drf0), self.z, self.omega, self.width, self.nz, r'$\Im{\chi^0}(\omega, z, z)$')
        else:
            show_map_omega(np.real(self.drf0), self.z, self.omega, self.width, self.nz, r'$\Re{\chi^0}(\omega, z, z)$')


    def show_map_drf(self, w, compl = False):
        if compl:
            show_map(np.imag(self.drf), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\chi}(z_1, z_2)$')
        else:
            show_map(np.real(self.drf), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\chi}(z_1, z_2)$')

    def show_map_drf_omega(self, compl = False):
        if compl:
            show_map_omega(np.imag(self.df0), self.z, self.omega, self.width, self.nz, r'$\Im{\chi}(\omega, z, z)$')
        else:
            show_map_omega(np.real(self.df0), self.z, self.omega, self.width, self.nz, r'$\Re{\chi}(\omega, z, z)$')

    def show_map_df(self, w, compl = False):
        if compl:
            show_map(np.imag(self.df), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Im{\epsilon}(z_1, z_2)$')
        else:
            show_map(np.real(self.df), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\epsilon}(z_1, z_2)$')

    def show_map_df_omega(self, compl = False):
        if compl:
            show_map_omega(np.imag(self.df), self.z, self.omega, self.width, self.nz, r'$\Im{\epsilon}(\omega, z, z)$')
        else:
            show_map_omega(np.real(self.df), self.z, self.omega, self.width, self.nz, r'$\Re{\epsilon}(\omega, z, z)$')

    def show_map_df_inv(self, w, compl = False):
        if compl:
            show_map(-np.imag(self.df_inv), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$-\Im{\epsilon^{-1}}(z_1, z_2)$')
        else:
            show_map(np.real(self.df_inv), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{\epsilon^{-1}}(z_1, z_2)$')

    def show_map_df_inv_omega(self, compl = False):
        if compl:
            show_map_omega(-np.imag(self.df_inv), self.z, self.omega, self.width, self.nz, r'$-\Im{\epsilon^{-1}}(\omega, z, z)$')
        else:
            show_map_omega(np.real(self.df_inv), self.z, self.omega, self.width, self.nz, r'$\Re{\epsilon^{-1}}(\omega, z, z)$')

    
    def show_map_scr_int(self, w, compl = False):
        if compl:
            show_map(-np.imag(self.scr_int), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$-\Im{W}(z_1, z_2)$')
        else:
            show_map(np.real(self.scr_int), self.z, self.omega, self.width, self.nz, self.nz_slab, w, r'$\Re{W}(z_1, z_2)$')

    def show_map_scr_int_omega(self, compl = False):
        if compl:
            show_map_omega(-np.imag(self.scr_int), self.z, self.omega, self.width, self.nz, r'$-\Im{W}(\omega, z, z)$')
        else:
            show_map_omega(np.real(self.scr_int), self.z, self.omega, self.width, self.nz, r'$\Re{W}(\omega, z, z)$')

    def show_loss(self):
        fig = go.Figure()
        xtitle = r"$\omega \text{ [eV]}$"
        fig.add_trace(go.Scatter(x = self.omega*27.211, y = np.real(self.loss), name = r"$V_{tot}$", mode = "lines+markers", marker=dict(
            size=5,
            ),))
        fig.update_layout(title_text = r'Loss function', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$L(\omega)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()

    def show_eig_modes(self, n_mode):
        if type(n_mode) == int:
            n_mode = range(n_mode)
        
        fig = go.Figure()
        xtitle = r"$\omega \text{ [eV]}$"
        for i in n_mode:
            fig.add_trace(go.Scatter(x = self.omega*27.211, y = -np.imag(np.power(self.eig_q[:, i], -1)), name = r"mode "+str(i), mode = "lines+markers", marker=dict(
                size=5,
                ),))
        fig.update_layout(title_text = r'Loss function', title_x=0.5,xaxis_title= xtitle,
                yaxis_title = r'$L(\omega)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        #fig_loss.update_yaxes(type="log")
        fig.show()
    
    def show_spat_mode(self, w, i):
        fig = go.Figure()
        xtitle = r"$z \text{[Bohr]}$"
        ytitle = r"$V_i(z), \rho_i(z)$"
        title = r"$\text{Eigenfunctions associated with the visible mode in the EELS spectra}$"
        y1 = np.real(np.fft.fft(self.vec[w, :, i]))
        y2 = np.real(np.fft.ifft(self.vec_dual[w, i, :], norm  = "forward"))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y = y1 , name = r"$V_i(z)$", mode = "lines",marker=dict(
            size=5,color = "blue",
            ),))
        fig.add_trace(go.Scatter(x = tools.center_z(self.z), y =  y2, name = r"$\rho_i(z)$", mode = "lines",marker=dict(
            size=5,color = "red",
            ),))
        fig.add_trace(go.Scatter(x=np.ones(100)*self.width/2,y= np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2))*1.05, 100), mode = "lines",line=dict(dash = "dot", color = "black", width  = 0,
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x=-np.ones(100)*self.width/2,y= np.linspace(min(min(y1), min(y2)), max(max(y1), max(y2))*1.05, 100), mode = "lines",line=dict(dash = "dot", color = "black", width  = 0,
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x = np.array([self.width/2, self.width/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))
        
        fig.add_trace(go.Scatter(x = np.array([-self.width/2, -self.width/2]), y =  np.array([min(min(y1), min(y2)), max(max(y1), max(y2))])*1.05, name = r"$\rho_i(z)$", mode = "lines",line=dict(dash = "dot",
            color = "black",
            ), showlegend = False))

        fig.update_layout(title={
                'text': title,
                'y':0.87,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'}, title_x=0.5, width=1000,
            height=600,xaxis_title= xtitle,
            yaxis_title = ytitle, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', legend=dict(bordercolor = "black", borderwidth = 1, x = 0.87, y = 0.7))
        fig.update_xaxes(color = "black", mirror="ticks", showline = True, visible = True, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside")
        fig.update_yaxes(color = "black", mirror="ticks", showline = True, visible = True, linewidth=1, linecolor='black',tickwidth=1, tickcolor='black', ticklen=10, nticks = 10,ticks="inside")

        fig.show()

    
    




