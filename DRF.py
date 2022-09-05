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
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
#mport plotly.graph_object as go


#Specifications: Starts from 2 files and a size of supercell (nc: bulk and slab have the same lateral dimensions) 
#Step 1: Build the matrix \chi0(q+G, q+G') from symmetries for all frequencies
#Step 2: Make the Fourier transform to get \chi0(\omega, r, r')
#Step 3: Reduce the size of the matrix by considering only the response along axe
#Step 4: Make the inverse Fourier transform 
#Step 5: Delivers the plasmon band structure

####Test functions####

def Friedel(x, per, amp, exp, delta, deph):
    return amp*np.cos(per*x+deph)/x**exp+delta

def Friedel_without_per(x, amp, exp, delta, deph):
    return amp*np.cos(2*math.pi/1.4*x+deph)/x**exp+delta 


def Gaussienne(x, sigma, mu = 0):
    return 1/(sigma*math.sqrt(2*math.pi))*np.exp(-(x-mu)**2/(2*sigma**2))

def func_test(x, a = 1):
    return np.exp(-abs(x)*a)


########## Figures #########

def OneGraph(X, Y, Name, Title, xaxis, yaxis):
    fig = go.Figure()
    xtitle = xaxis
    fig.add_trace(go.Scatter(x = X, y = Y, name = Name, mode = "lines+markers",marker=dict(
                size=5,
            ),))
    fig.update_layout(title_text = Title, title_x=0.5,xaxis_title= xtitle,
            yaxis_title = yaxis,paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',)
    fig.show(dpi = 300) 

def center_z(z):
    n = len(z)
    z_half = max(z)/2
    return np.linspace(-z_half, z_half, n)



########## Jellium 2nd Principle ###########

def SecondP_jellium(qmax, dens, omega, n = 0.025):
    chi0wzz_B = chi0wzz_jellium(qmax, dens, omega)
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    z = qvec_to_zvec(q, dens)
    step = abs(z[1]-z[0])
    d = 70
    nz = (round(d/step))
    d_eff = step*nz
    z1, z2 = np.linspace(0, d_eff, nz)
    nband = 700
    chiwzz_S = chi0wzz_slab_jellium_Eguiluz_1step(z1, z2, omega, n, d, nband)

    return chi0wzz

########## Jellium Bulk 2 components ##########


def chi0wzz_jellium(qmax, dens, omega, n = 0.025):
    nw = len(omega)
    nz = 2*qmax*dens+1
    chi0wz = np.zeros((nw, nz), dtype = complex)
    chi0wzz = np.zeros((nw, nz, nz), dtype = complex)
    for i in range(nw):
        chi0wz[i, :], z = chi0_r_1d(qmax, dens, omega[i], n = 0.025)
        chi0wzz[i, 0, :] = chi0wz[i, :]
    for i in range(nw):
        for j in range(1, nz):
            chi0wzz[i, j, :] = np.append(chi0wz[i, nz-j:nz], chi0wz[i, 0:nz-j])
    return chi0wzz

########## Jellium 1D #############
def chi0_q_1d(qmax, dens, omega, n = 0.025):
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    chi0_q_1d = np.zeros((nq), dtype = complex)
    qrev = Rev_vec(q)
    for i in range(nq):
        qi = qrev[i]**2
        norm_q = (qi)**(1/2)
        epsilon_q_1d = epsilon_1(norm_q, omega, n)+ic*epsilon_2(norm_q, omega, n)
        chi0_q_1d[i] = -(epsilon_q_1d-1)*e0*norm_q**2
    kF = (3*math.pi**2*n)**(1/3)
    #if omega == 0:
    #    chi0_q_1d[0] = -4*kF/(4*math.pi**2)
    return chi0_q_1d

def chi0_r_1d(qmax, dens, omega, n = 0.025):
    chi0_q = chi0_q_1d(qmax, dens, omega)
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    z = qvec_to_zvec(q, dens)
    chi0z = np.fft.ifftn(chi0_q)
    size = chi0z.size
    chi0z = chi0z*size/max(z)
    return chi0z, z
########## Jellium 2D #############
def chi0_q_2d(qmax, dens, omega, n = 0.025):
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    chi0_q_2d = np.zeros((nq, nq), dtype = complex)
    qrev = Rev_vec(q)
    for i in range(nq):
        qi = qrev[i]**2
        for j in range (nq):
            qj = qrev[j]**2
            norm_q = (qi + qj)**(1/2)
            epsilon_q_2d = epsilon_1(norm_q, omega, n)+ic*epsilon_2(norm_q, omega, n)
            chi0_q_2d[i, j] = -(epsilon_q_2d-1)*e0*norm_q**2
    kF = (3*math.pi**2*n)**(1/3)
    if omega ==0 :
        chi0_q_2d[0, 0] = -4*kF/(4*math.pi**2)
    return chi0_q_2d

def chi0_r_2d(qmax, dens, omega, n = 0.025):
    chi0_q = chi0_q_2d(qmax, dens, omega)
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    z = qvec_to_zvec(q, dens)
    chi0z = np.fft.ifftn(chi0_q)
    size = chi0z.size
    chi0z = chi0z*size/max(z)**2
    return chi0z, z

########## Jellium 3D #############
def chi0_q_3d(qmax, dens, omega, n = 0.025):
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    chi0_q_3d = np.zeros((nq, nq, nq), dtype = complex)
    qrev = Rev_vec(q)
    for i in range(nq):
        qi = qrev[i]**2
        for j in range (nq):
            qj = qrev[j]**2
            for k in range(nq):
                qk = qrev[k]**2
                norm_q = (qi + qj + qk)**(1/2)
                epsilon_q_3d = epsilon_1(norm_q, omega, n)+ic*epsilon_2(norm_q, omega, n)
                chi0_q_3d[i, j, k] = -(epsilon_q_3d-1)*e0*norm_q**2
    kF = (3*math.pi**2*n)**(1/3)
    if omega == 0:
        chi0_q_3d[0, 0, 0] = -4*kF/(4*math.pi**2)
    return chi0_q_3d
    
def chi0_r_3d(qmax, dens, omega, n = 0.025):
    chi0_q = chi0_q_3d(qmax, dens, omega)
    nq = 2*dens*qmax+1
    q = np.linspace(-qmax, qmax, nq)
    z = qvec_to_zvec(q, dens)
    chi0z = np.fft.ifftn(chi0_q)
    size = chi0z.size
    chi0z = chi0z*size/max(z)**3
    return chi0z, z



########## FFT Tools ##############
def zvec_to_qvec(z, dens):
    nz = len(z)
    q_out = math.pi*dens
    q = np.linspace(-q_out, q_out, nz)
    return q

def qvec_to_zvec(q, dens):
    nq = len(q)
    z_out = 2*math.pi*dens
    z = np.linspace(0, z_out, nq)
    return z

########## Slab Jellium ##############
ic = complex(0,1)
def F_ll_Despoja2005(q_p, omega, l1, l2, d, EF):
    eta = 1e-10
    a = a_ll(q_p, l1, l2, d)
    k = k_l2(l1, d, EF)
    return -1/(math.pi*q_p**2)*(2*a+np.sign(omega-a)*cmath.sqrt((omega - a +ic*eta)**2-q_p**2*k)-np.sign(omega+a)*cmath.sqrt((omega + a +ic*eta)**2-q_p**2*k))


    
def a_ij(q_p, ei, ej):
    return (q_p**2)/2-(ei-ej)
def k_i(ei, d, EF):
    return (2*(EF-ei))**(1/2)

def k_l2(l, d, EF):
    return (2*(EF-el(l, d)))
def el(l, d):
    return 1/2*l**2*math.pi**2/d**2
def a_ll(q_p, l1, l2, d):
    return (q_p**2)/2-(el(l1, d)-el(l2, d))



def F_ll(q_p, omega, l1, l2, d, EF, eta):
    ic = complex(0, 1)
    n=0.025
    a = a_ll(q_p, l1, l2, d)
    kF = (3*math.pi**2*n)**(1/3)
    EF = (1/2)*(3*math.pi**2*n)**(2/3)
    nmax = math.ceil(d*kF/math.pi)
    prefactor = (EF-el(l1, d))/(math.pi)
    return -prefactor*(1/(a+omega+ic*eta)+1/(a-omega-ic*eta))

def chi0wzz_slab_jellium_Eguiluz_1step_F(z1, z2, omega, n, d, eta):
    nw = len(omega)
    nz1, nz2 = len(z1), len(z2)
    kF = (3*math.pi**2*n)**(1/3)
    EF = (1/2)*(3*math.pi**2*n)**(2/3)
    nmax = math.ceil(d*kF/math.pi)
    nband = nmax*4
    chi0wzz = np.zeros((nw, nz1, nz2), dtype = complex)
    energies = np.zeros((nband))
    for i in range(1, nband):
        energies[i] = 1/2*(i**2*math.pi**2)/d**2
    wf1 = np.zeros((nz1, nband))
    alpha_band = np.zeros(nband)
    for j in range(1, nband):
        alpha_band[j] = math.pi*j/d
        wf1[:, j] = np.sin(alpha_band[j]*z1)
    wf2 = np.zeros((nz2, nband))
    for j in range(1, nband):
        wf2[:, j] = np.sin(alpha_band[j]*z2)
    wff = np.zeros((nz1, nz2, nband))
    for i in range(nz1):
        for j in range(nz2):
            for k in range(1, nband):
                wff[i, j, k] = wf1[i, k]*wf2[j,k]
    wff = (2/d)*wff
    for i in range(nw):
        print("omega = "+str(omega[i]))
        Fll = np.zeros((nmax, nband), dtype = complex)
        for j in range(1, nmax):
            for k in range(1, nband):
                Fll[j, k] = F_ll(0, omega[i], j, k, d, EF, eta)
        for j in range(nz1):
            for k in range(nz2):
                for l in range(1, nmax):
                    wffi = wff[j, k, l]
                    for m in range(1, nband):
                        if l == m:
                            continue
                        wffj = wff[j, k, m]
                        chi0wzz[i, j, k]+=wffi*wffj*Fll[l, m]               
    return chi0wzz

def chi0wzz_slab_jellium_Eguiluz_1step(z1, z2, omega, n, d, nband):
    nw = len(omega)
    nz1, nz2 = len(z1), len(z2)
    kF = (3*math.pi**2*n)**(1/3)
    EF = (1/2)*(3*math.pi**2*n)**(2/3)
    nmax = math.ceil(d*kF/math.pi)
    chi0wzz = np.zeros((nw, nz1, nz2), dtype = complex)
    energies = np.zeros((nband))
    for i in range(1, nband):
        energies[i] = 1/2*(i**2*math.pi**2)/d**2
    wf1 = np.zeros((nz1, nband))
    alpha_band = np.zeros(nband)
    for j in range(1, nband):
        alpha_band[j] = math.pi*j/d
        wf1[:, j] = np.sin(alpha_band[j]*z1)
    wf2 = np.zeros((nz2, nband))
    for j in range(1, nband):
        wf2[:, j] = np.sin(alpha_band[j]*z2)
    wff = np.zeros((nz1, nz2, nband))
    for i in range(nz1):
        for j in range(nz2):
            for k in range(1, nband):
                wff[i, j, k] = wf1[i, k]*wf2[j,k]
    wff = 2/d*wff
    for i in range(nw):
        print("omega = "+str(omega[i]))
        Fll = np.zeros((nmax, nband))
        for j in range(1, nmax):
            alpha = 2*(EF-energies[j])
            for k in range(1, nband):
                if j == k:
                    continue
                else:
                    Fll[j, k] = (energies[k]-energies[j])/((energies[j]-energies[k])**2-omega[i]**2)
            Fll[j, :] = -alpha*Fll[j, :]/math.pi
        for j in range(nz1):
            for k in range(nz2):
                for l in range(1, nmax):
                    wffi = wff[j, k, l]
                    for m in range(1, nband):
                        if l == m:
                            continue
                        wffj = wff[j, k, m]
                        chi0wzz[i, j, k]+=wffi*wffj*Fll[l, m]               
    return chi0wzz










def wave_func_n_vec(z, band, l):
    alpha = band*math.pi/l
    return math.sqrt(2/l)*np.sin(alpha*z)

def wave_func_n(z, alpha, l):
    """  if z<0 or z>l:
        return 0  """
    return math.sqrt(2/l)*math.sin(alpha*z)

def epsilon_z_slab(z, lslab, omega, n = 0.025):
    if omega == 0:
        return epsilon_z_slab(z, lslab, 0.01, n)
    sum_func = 0
    kF = (3*math.pi**2*n)**(1/3)
    d = 3*math.pi/(8*kF) + math.pi**2/(8*kF**2*lslab)
    l = lslab+2*d
    wp = math.sqrt(n/(e0))
    nmax = round(l*kF/math.pi)
    for i in range(nmax):
        alpha = math.pi*i/l
        wf = wave_func_n(z, alpha, lslab)
        dens = wf**2
        sum_func+=(kF**2-alpha**2)*dens
    return 1 - wp**2/(2*math.pi*n*omega**2)*sum_func


###Project function###

######Lindhard###########
e0 =  1/(4*math.pi)
def dchi0q(q, n = 0.025):
    kF = (3*math.pi**2*n)**(1/3)
    return -4*kF/(4*math.pi**2)*kF/2*(1/(q*kF)-math.log(abs((2*kF+q)/(2*kF-q)))*(1/q**2+1/(4*kF**2)))

def fchi0q(q, n = 0.025):
    kF = (3*math.pi**2*n)**(1/3)
    return -4*kF/(4*math.pi**2)*(1/2+kF/(2*q)*math.log(abs((2*kF+q)/(2*kF-q)))*(1-(q**2)/(4*kF**2)))

def pol3(x, a, b, c, d):
    return a*x**3+b*x**2+c*x+d

def chi0qcutoff(qlim, qmax, dens, omega, n = 0.025):
    if qmax < qlim:
        raise ValueError("qmax ("+str(qmax)+") must be larger than qlim ("+str(qlim)+")")
    npoints = qmax*dens+1
    qlim_ind = round(qlim*dens/2)
    q = np.linspace(-qmax, qmax, npoints, endpoint = True)
    q_rev = -np.real(Rev_vec(q))
    q_rev[0] = 0
    kF = (3*math.pi**2*n)**(1/3)
    epsq = eps(q_rev, omega, n)
    nq, nw = epsq.shape
    const = -2*kF/(4*math.pi**2)
    chi0q = np.zeros((nq, nw), dtype = complex)
    for omega in range(nw):
        for qvec in range(nq):
            if abs(q_rev[qvec])==2*kF and omega == 0:
                print("IN")
                chi0q[qvec, omega] = 1/2*const
            else:
                chi0q[qvec, omega] = -(epsq[qvec, omega]-1)*e0*q_rev[qvec]**2

    chi0q[0, 0] = const
    A = [[3*qlim**2, 2*qlim, 1, 0], [3*qmax**2, 2*qmax, 1, 0], [qlim**3, qlim**2, qlim, 1], [qmax**3, qmax**2, qmax, 1]]
    b = [dchi0q(qlim, n), 0, fchi0q(qlim), 0]
    coef = np.linalg.solve(A, b)
    
    for i in range(nw):
        for j in range(qlim_ind, nq - qlim_ind):
            chi0q[j, i] = pol3(abs(q_rev[j]), coef[0], coef[1], coef[2], coef[3])
    for i in range(omega):
        chi0q[:, i] = Rev_vec(chi0q[:, i])

    return chi0q/2, q

def chi0z_cutoff(chi0q, dens):
    nq, nw = chi0q.shape
    chi0z = np.zeros((nq, nw), dtype = complex)
    for i in range(nw):
        chi0z[:, i] = np.fft.ifft(Rev_vec(chi0q[:, i]))
    z_out = 2*math.pi*dens
    z = np.linspace(0, z_out, nq)
    chi0z_out = chi0z*nq/z_out
    return chi0z_out, z

def Iq(q, n=0.025):
    kF = (3*math.pi**2*n)**(1/3)
    #q = abs(q)
    if q == 0:
        return 1
    else:
        return 1/2+kF/(2*q)*math.log(abs((2*kF+q)/(2*kF-q)))*(1-(q**2)/(4*kF**2))

def epsilon_1(q, omega, n = 0.025):
    q = abs(q)
    kF = (3*math.pi**2*n)**(1/3)
    EF = (1/2)*(3*math.pi**2*n)**(2/3)
    pF = math.sqrt(2*EF)
    vF = pF
    wp = math.sqrt(n/(e0))
    kTF = math.sqrt(4*(3*n/math.pi)**(1/3))
    #print(EF, kF, pF, wp, kTF)
    if q==0 and omega != 0:
        return 1-wp**2/omega**2
    elif q==0 and omega==0:
        return epsilon_1(q, 0.01, n)
    
    else:
        eq = (q**2)/2
        ln1 = math.log(abs((eq+q*vF+omega)/(eq-q*vF+omega)))
        ln2 = math.log(abs((eq+q*vF-omega)/(eq-q*vF-omega)))
        pre = 1/(2*kF*q**3)
        return 1+((kTF**2)/(2*q**2))*(1+pre*(4*EF*eq-(eq+omega)**2)*ln1+pre*(4*EF*eq-(eq-omega)**2)*ln2)

def epsilon_2(q, omega, n = 0.025):
    kF = (3*math.pi**2*n)**(1/3)
    EF = (1/2)*(3*math.pi**2*n)**(2/3)
    pF = math.sqrt(2*EF)
    vF = pF
    wp = math.sqrt(n/e0)
    kTF = math.sqrt(4*(3*n/math.pi)**(1/3))
    q = abs(q)
    eq = q**2/(2)
    if abs(q)<2*kF:
        if abs(omega)>0 and abs(omega)<(abs(q)*vF-eq):
            return 2*omega/q**3
        elif abs(omega)>= abs(q)*vF-eq and abs(omega)<= eq+abs(q)*vF:
            return (1/q**3)*(kF**2-(1/q)**2*(omega-eq)**2)
        else:
            return 0
    else:
        if abs(omega)>= abs(q)*vF-eq and abs(omega)<= eq+abs(q)*vF:
            return (1/q**3)*(kF**2-(1/q)**2*(omega-eq)**2)
        else:
            return 0


def eps(q, omega, n = 0.025):
    nq = len(q)
    nw = len(omega)
    eps = np.zeros((nq, nw), dtype = complex)
    ic = complex(0,1)
    for i in range(nq):
        for j in range(nw):
            eps[i, j] = epsilon_1(q[i], omega[j], n) + ic* epsilon_2(q[i], omega[j], n)
    return eps

def chi0q(q, omega, n = 0.025):
    kF = (3*math.pi**2*n)**(1/3)
    epsq = eps(q, omega, n)
    nq, nw = epsq.shape
    const = -4*kF/(4*math.pi**2)
    chi0q = np.zeros((nq, nw), dtype = complex)
    for omega in range(nw):
        for qvec in range(nq):
            if abs(q[qvec])==2*kF and omega == 0:
                print("IN")
                chi0q[qvec, omega] = 1/2*const
            else:
                chi0q[qvec, omega] = -(epsq[qvec, omega]-1)*e0*q[qvec]**2
    chi0q[0, 0] = const
    return chi0q/2  

def chi0q_nolim(q, omega, n = 0.025):
    kF = (3*math.pi**2*n)**(1/3)
    epsq = eps(q, omega, n)
    nq, nw = epsq.shape
    const = -4*kF/(4*math.pi**2)
    chi0q = np.zeros((nq, nw), dtype = complex)
    for omega in range(nw):
        for qvec in range(nq):
            if abs(q[qvec])==2*kF and omega ==0:
                print("IN")
                chi0q[qvec, omega] = 1/2*const
            else:
                chi0q[qvec, omega] = -(epsq[qvec, omega]-1)*e0*q[qvec]**2
    return chi0q/2  

def chi0z(q, omega, n = 0.025):
    chi0 = chi0q(q, omega, n)
    nq, nw = chi0.shape
    z_out = 2*math.pi/q[1]
    z = np.linspace(0, z_out, nq)
    chi0z = np.zeros((nq, nw), dtype = complex)
    for i in range(nw):
        chi0z[:, i]  = np.fft.ifft(chi0[:, i])
    chi0z_out = chi0z*nq/z_out
    return chi0z_out, z

def chi0z_nolim(q, omega, n = 0.025):
    chi0 = chi0q_nolim(q, omega, n)
    nq, nw = chi0.shape
    z_out = 2*math.pi/q[1]
    z = np.linspace(0, z_out, nq)
    chi0z = np.zeros((nq, nw), dtype = complex)
    for i in range(nw):
        chi0z[:, i]  = np.fft.ifft(chi0[:, i])
    chi0z_out = chi0z*nq/z_out
    return chi0z_out, z


######Post-Process Abinit########

def centerslab(chi0zzS):
    d1, d2 = chi0zzS.shape
    chi0zzS_centered = np.zeros((d1, d2), dtype = complex)
    for i in range(d1):
        chi0zzS_centered[i] = Rev_vec(chi0zzS[i])
        chi0zzS_centered[i] = chi0zzS_centered[i][::-1]
    chi0_out = np.zeros((d1, d2), dtype = complex)
    for i in range(d2):
        chi0_out[:, i] = Rev_vec(chi0zzS_centered[:,i])
        chi0_out[:, i] = chi0_out[:, i][::-1]
    return chi0_out


def model1(chi0zzS, chi0zzB, cut_off, scale, void):
    #Attention:first point not equal to last point
    d1, d2 = chi0zzS.shape
    n1, n2 = chi0zzB.shape
    chi0zzB_sym = chi0zz_SYM(chi0zzB)
    Chi0zzS_cent = centerslab(chi0zzS)
    thickness = d2+n2
    Chi0 = np.zeros((thickness, thickness), dtype = complex)
    if d2%2==0:
        slab_lim_left = math.floor(d2/2)
    else:
        slab_lim_left = math.floor(d2/2)+1
    slab_lim_right = thickness - slab_lim_left
    for i in range(thickness):
        for j in range(thickness):
            if (i < slab_lim_left and j < slab_lim_left):
                Chi0[i, j] = Chi0zzS_cent[i, j]
            elif (i > slab_lim_right and j > slab_lim_right):
                Chi0[i, j] = Chi0zzS_cent[i-n2, j-n2]
            else:
                Chi0[i, j] = chi0zzB_sym[i%n2, j%n2]
    #cut_off = min(n2, d2)
    for i in range(thickness):
        for j in range(thickness):
            if i < void or i > thickness - void or j < void or j > thickness - void:
                Chi0[i,j] = 0
            elif abs(i-j)*scale>cut_off:
                Chi0[i,j] = 0
    return Chi0

def model10(chi0zzS, chi0zzB):
    #Attention:first point not equal to last point
    d1, d2 = chi0zzS.shape
    n1, n2 = chi0zzB.shape
    chi0zzB = CenterChi0(chi0zzB)
    chi0zzS = CenterChi0(chi0zzS)
    thickness = d2+n2
    if d2%2==0:
        u1 = math.floor(d2/2)
    else:
        u1 = math.floor(d2/2)+1
    u2 = thickness - u1
    chi0zz = np.zeros((thickness, thickness), dtype = complex)
    Chi0zzB = np.zeros((thickness, thickness), dtype = complex)
    Chi0zzS = np.zeros((thickness, thickness), dtype = complex)
    Chi0 = np.zeros((thickness, thickness), dtype = complex)
    for i in range(u1):
        Chi0zzS[i, 0:d2] = chi0zzS[i]
    count = 0
    for i in range(u2,thickness):
        Chi0zzS[i, thickness - d2:thickness] = chi0zzS[math.floor(d2/2)+count]
        count+=1
    #print(thickness)
    #print(n2)
    #print(math.floor(n2/2))
    if n1%2==0:
        r1 = math.floor(n1/2)
    else:
        r1 = math.floor(n1/2)+1
    if n2%n2==0:
        r2 = math.floor(n2/2)
    else:
        r2 = math.floor(n2/2)+1
    for i in range(r2):
        j = (math.floor((i+r1)/n1))*n1
        l1= thickness + j - r2
        l2= r2 + j
        Chi0zzB[i, l1:thickness] = chi0zzB[(i+r1)%n1, 0:-j+r2]
        Chi0zzB[i, 0:l2] = chi0zzB[(i+r1)%n1, -j+r2:n2]
    for i in range(r2, thickness - r2):
        ind1 = math.floor((i+r1)/n1)*n1-r2
        ind2 = math.floor((i+r1)/n1)*n1+r2
        Chi0zzB[i, ind1:ind2] = chi0zzB[(i+r1)%n1]
    count = 0
    for i in range(thickness - r2, thickness):
        j = math.floor((count+r1)/n1)*n1
        l1= thickness + j - n2
        l2= j
        Chi0zzB[i, l1:thickness] = chi0zzB[(i+r1)%n1, 0:-j+n2]
        Chi0zzB[i, 0:l2] = chi0zzB[(i+r1)%n1, -j+n2:n2]
        count +=1
    for i in range(thickness):
        for j in range(thickness):
            if (i < u1 and j < u1) or (i>u2 and j>u2):
                Chi0[i,j] = Chi0zzS[i,j]
            else:
                Chi0[i,j] = Chi0zzB[i,j]
    return Chi0zzB, Chi0zzS, Chi0


def chi0wzzsmall(filename, axe = "001"):
    sus_ncfile = ScrFile(filename)
    nw = sus_ncfile.reader.nw
    structure=abipy.core.structure.Structure.from_file(filename)
    lattice = structure.lattice.matrix
    A, B, C = lattice[0], lattice[1], lattice[2]
    vol = np.dot(A, (np.cross(B, C)))
    chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0omegaGG_fromSym(filename, nw)
    chi0rr = FFT_chi0_from_Mat(chi0GG[0], ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol)
    chi0rr0 = chi0rr
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    chi0zz0 = chi0zz(chi0rr, axe)
    d1, d2 = chi0zz0.shape
    chi0wzz = np.zeros((nw, d1, d2), dtype = complex)
    chi0wzz[0] = chi0zz0
    for i in range(1, nw):
        chi0rr = FFT_chi0_from_Mat(chi0GG[i], ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol)
        chi0wzz[i] = chi0zz(chi0rr, axe)
    return chi0wzz


def chi0wzz(filenameS, filenameB, opt = "FromSym", opt2 = "Kaltak", axe = "001"):
    start_time = time.time()
    #Collect generic information in order to verify the validity (only for freq) of the input files
    sus_ncfileS = ScrFile(filenameS)
    sus_ncfileB = ScrFile(filenameB)
    #Number of Freq of the input files
    nwS = sus_ncfileS.reader.nw
    nwB = sus_ncfileB.reader.nw
    if nwS != nwB:
        raise ValueError("nomega must be the same for the slab and the bulk. Please provide files with the same frequency sampling (actual sampling gives nwS="+str(nwS)+" and nwB="+str(nwB)+")")
    #Build chi0(\omega, q+G, q+G')
    structureS=abipy.core.structure.Structure.from_file(filenameS)
    latticeS = structureS.lattice.matrix
    As, Bs, Cs = latticeS[0], latticeS[1], latticeS[2]
    vols = np.dot(As, (np.cross(Bs, Cs)))
    structureB=abipy.core.structure.Structure.from_file(filenameB)
    latticeB = structureB.lattice.matrix
    Ab, Bb, Cb = latticeB[0], latticeB[1], latticeB[2]
    volb = np.dot(Ab, (np.cross(Bb, Cb)))
    ncell = round(vols/volb)

    chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0omegaGG_fromSym(filenameB, nwB)
    chi0rr = FFT_chi0_from_Mat(chi0GG[0], ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol)
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    #print(chi0rr[0, 0, 0, 0, 0, 0])
    chi0zzB = chi0zz(chi0rr, axe)
    d1, d2 = chi0zzB.shape
    chi0wzzB = np.zeros((nwB, d1, d2), dtype = complex)
    chi0wzzB[0] = chi0zzB
    for i in range(1, nwS):
        chi0rr = FFT_chi0_from_Mat(chi0GG[i], ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol)
        chi0wzzB[i] = chi0zz(chi0rr, axe)
    #print(chi0wzzB.shape)
    l1, l2, l3 = n4, n5, n3*ncell
    chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0omegaGG_fromSym(filenameS, nwS)
    
    chi0rr = FFT_chi0_sizeadapt(chi0GG[0], ind_qbzG_to_vec, l1, l2, l3, l1, l2, l3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2 = "Kaltak")
    #print(chi0rr.shape)
    #print(chi0rr[0, 0, 0, 0, 0, 0])
    chi0zzS = chi0zz(chi0rr, axe)
    d1, d2 = chi0zzS.shape
    chi0wzzS = np.zeros((nwS, d1, d2), dtype = complex)
    chi0wzzS[0] = chi0zzS
    for i in range(1, nwS):
        chi0rr = FFT_chi0_sizeadapt(chi0GG[i], ind_qbzG_to_vec, l1, l2, l3, l1, l2, l3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2 = "Kaltak")
        chi0wzzS[i] = chi0zz(chi0rr, axe)
    #print(chi0wzzS.shape)
    Lchi0zz = LargeChi0(chi0wzzB[0], chi0wzzS[0])
    s1, s2 = Lchi0zz.shape
    Chi0wzz = np.zeros((nwS, s1, s2), dtype = complex)
    Chi0wzz[0] = Lchi0zz
    for i in range(1,nwS):
        Chi0wzz[i] = LargeChi0(chi0wzzB[i], chi0wzzS[i])
    elapsed2 = time.time()-start_time
    print("The whole second principle computation took "+str(elapsed2/60)+" minutes")
    return Chi0wzz

def chi0zz(chi0rr, axe = "001"):
    d1, d2, d3, d4, d5, d6 = chi0rr.shape
    if axe == "001":
        chi0zz = np.zeros((d3, d6), dtype = complex)
        for i in range(d3):
            for j in range(d6):
                chi0zz[i, j]=np.sum(chi0rr[:, :, i, :, :, j])
        chi0zz= chi0zz/(d1*d2*d4*d5)
    if axe == "010":
        chi0zz = np.zeros((d2, d5), dtype = complex)
        for i in range(d2):
            for j in range(d5):
                chi0zz[i, j]=np.sum(chi0rr[:, i, :, :, j, :])
        chi0zz= chi0zz/(d1*d3*d4*d6)
    if axe == "100":
        chi0zz = np.zeros((d1, d4), dtype = complex)
        for i in range(d1):
            for j in range(d4):
                chi0zz[i, j]=np.sum(chi0rr[i, :, :, j, :, :])
        chi0zz= chi0zz/(d2*d3*d5*d6)
    chi0zz_out = chi0zz_SYM(chi0zz)
    return chi0zz

def chi0zz_SYM(chi0zz):
    d1, d2 = chi0zz.shape
    chi0zz_S0 = np.zeros((d2, d2), complex)
    chi0zz_S = np.zeros((d2, d2), complex)
    N = round(d2/d1)
    for i in range(d1):
        chi0zz_S0[i, :] = chi0zz[i, :]
        chi0zz_S0[:, i] = chi0zz[i, :]
    for i in range(N):
        ind1, ind2 = i*d1, d2-i*d1
        chi0zz_S[ind1:d2,ind1:d2] = chi0zz_S0[0:ind2, 0:ind2]
    return chi0zz_S


def chi0Comp(chi0rr, x, y, axe = "001"):
    d1, d2, d3, d4, d5, d6 = chi0rr.shape
    if axe == "001":
        chi0zz = np.zeros((d3, d6), dtype = complex)
        for i in range(d3):
            chi0zz[i, :] = chi0rr[x, y, i, x, y, :]
        return chi0zz
    if axe == "010":
        chi0zz = np.zeros((d2, d5), dtype = complex)
        for i in range(d2):
            chi0zz[i, :] = chi0rr[x, i, y, x, :, y]
        return chi0zz
    if axe == "001":
        chi0zz = np.zeros((d1, d4), dtype = complex)
        for i in range(d1):
            chi0zz[i, :] = chi0rr[i, x, y, :, x, y]
        return chi0zz

def CenterChi0(chi0zz):
    d1, d2 = chi0zz.shape
    count = 0
    chi0zz_out = np.zeros((d1, d2), dtype = complex)
    if d1%2 != 0:
        m = math.floor(d1/2)+1
    else:
        m = math.floor(d1/2)
    for i in range(m):
        chi0zz_out[i] = Rev_vec(chi0zz[i])
    for i in range(m, d1):
        resp = Rev_vec(chi0zz[math.floor(d1/2)-count-1, :])
        count+=1
        arr = resp[::-1]
        chi0zz_out[i] = arr
    for j in range(d2):
        chi0zz_out[:, j] = Rev_vec(chi0zz_out[:, j])
    return chi0zz_out

def CenterChi00(chi0zz):
    d1, d2 = chi0zz.shape
    count = 0
    if d1%2 != 0:
        chi0zz_out = np.zeros((d1, d2), dtype = complex)
        for i in range(d1):
            if i <= math.floor(d1/2):
                chi0zz_out[i, :] = Rev_vec(chi0zz[i, :])
            else:
                arr = Rev_vec(chi0zz[i-count-1, :])
                count+=2
                arr_out = arr[::-1]
                chi0zz_out[i, :] = arr_out
    else:
        chi0zz_out = np.zeros((d1, d2), dtype = complex)
        for i in range(d1):
            if i <= math.floor(d1/2):
                chi0zz_out[i, :] = Rev_vec(chi0zz[i, :])
            else:
                arr = Rev_vec(chi0zz[i-count-1, :])
                count+=2
                arr = arr[::-1]
                chi0zz_out[i, :] = arr
    for j in range(d2):
        chi0zz_out[:, j] = Rev_vec(chi0zz_out[:, j])
    return chi0zz_out

def Rev_vec(Y):
    #In : [0, 1, 2, 3, 3, 2, 1, 0] // [0, 1, 2, 3, 3, 2, 1]
    #Out : [3, 2, 1, 0, 0, 1, 2, 3] // [3, 2, 1, 0, 1, 2, 3]
    l = Y.size
    Y_out = np.zeros((l), dtype = complex)
    if l%2==0:
        m = math.floor(l/2)
    else:
        m = math.floor(l/2)+1
    Y_out[0:m] = Y[0:m][::-1]
    Y_out[m:l] = Y[m:l][::-1]
    return Y_out
        
#Build the matrix for the larger slab from second principles
def LargeChi0(chi0zzB, chi0zzS):
    #Model 1: set to 0 if no data available
    db1, db2 = chi0zzB.shape
    ds1, ds2 = chi0zzS.shape
    chi0zzB = CenterChi0(chi0zzB)
    chi0zzS = CenterChi0(chi0zzS)
    d = db2+ds2
    chi0zz = np.zeros((d, d), dtype = complex)
    for i in range(round(ds1/2)+1):
        chi0zz[i, 0:ds2] = chi0zzS[i, :]
    count = 0
    for i in range(round(ds1/2)+1, d-round(ds1/2)-1):
        a = round(ds1/2)+round(db1/2)+round(db1*((count-count%db1)/db1))-round(db2/2)
        b = a+db2
        chi0zz[i, a:b] = chi0zzB[count%db1-(math.floor(db1/2)), :]
        count+=1
    count = 1
    for i in range(d-round(ds1/2)-1,d):
        chi0zz[d-count, d-ds2:d] = chi0zzS[ds2-(i-d+round(ds1/2))-2, :]
        count+=1
    return chi0zz

def qibz_dict(kpoints, nk, nkpt):
    qibzvec_to_ind  = {}
    for i in range(nkpt):
        q = np.round(np.multiply(kpoints[i].frac_coords, nk))
        qibzvec_to_ind[(q[0], q[1], q[2])] = i
    return qibzvec_to_ind

def qbz_dict(SymRec, nsym, nkpt, kpoints, nk):
    vec_q_to_ind, ind_q_to_vec, sym_dict = {},{},{}
    ind = 0
    for i in range(nsym):
        for j in range(nkpt):
            q = np.round(np.matmul(SymRec[i], kpoints[j].frac_coords), 3)#+TRec[i]
            if np.amax(q) > 0.5 or np.amin(q) <= - 0.5: #if out of BZ ==> Umklapp
                a, b, c = q[0], q[1], q[2]
                if q[0]>0.5:
                    a=round(q[0]-1,3)
                elif q[0]<=-0.5:
                    a=round(q[0]+1,3)
                if q[1]>0.5:
                    b=round(q[1]-1,3)
                elif q[1]<=-0.5:
                    b=round(q[1]+1,3)
                if q[2]>0.5:
                    c=round(q[2]-1,3)
                elif q[2]<=-0.5:
                    c=round(q[2]+1,3)
                q_in_bz = (a, b, c)
            else:
                q_in_bz = (q[0], q[1], q[2])
            q_vec = np.round(np.multiply([q_in_bz[0], q_in_bz[1], q_in_bz[2]], nk))
            if (q_vec[0], q_vec[1], q_vec[2]) not in vec_q_to_ind.keys():
                        ind_q_to_vec[ind] = (q_vec[0], q_vec[1], q_vec[2])
                        vec_q_to_ind[(q_vec[0], q_vec[1], q_vec[2])] = ind
                        # Dict linking the index of the nth qpoint to the qibz and the symmetry required to obtain it
                        sym_dict[ind] = (i, j, (0, 0))
                        ind+=1
            else:
                continue
    return vec_q_to_ind, ind_q_to_vec, sym_dict

def IsInvSymIn(SymRec, nsym):
    Invlist=((-1,0,0),(0,-1,0),(0,0,-1))
    SymDict={}
    for i in range(nsym):
        SymRot=SymRec[i]
        Sym=((SymRot[0,0],SymRot[0,1],SymRot[0,2]),(SymRot[1,0],SymRot[1,1],SymRot[1,2]),(SymRot[2,0],SymRot[2,1],SymRot[2,2]))
        SymDict[Sym]=i
    if Invlist in SymDict.keys():
        return True
    else:
        return False

def add_invsym(vec_q_to_ind, ind_q_to_vec, sym_dict, nsym):
    ind=0
    for i in range(len(vec_q_to_ind)):
        q=ind_q_to_vec[i]
        if (-q[0], -q[1], -q[2]) not in vec_q_to_ind.keys():
            vec_q_to_ind[(-q[0], -q[1], -q[2])] = ind
            ind_q_to_vec[ind] = (-q[0], -q[1], -q[2])
            qsym = (sym_dict[i][0], sym_dict[i][1])
            sym_dict[ind] = (nsym+1, i, qsym)
            ind += 1
        else:
            continue
    return vec_q_to_ind, ind_q_to_vec, sym_dict

def G_dict(G, ng, nk):
    vec_G_to_ind, ind_G_to_vec = {}, {}
    for i in range(ng):
        G_vec = np.round(np.multiply([G[i, 0], G[i, 1], G[i, 2]], nk))
        vec_G_to_ind[(G_vec[0], G_vec[1], G_vec[2])] = i
        ind_G_to_vec[i] = (G_vec[0], G_vec[1], G_vec[2])
    return vec_G_to_ind, ind_G_to_vec

def qGibz_dict(nkpt, kpoints, ng, G, nk):
    vec_qibzG_to_ind, ind_qibzG_to_vec = {}, {}
    for i in range(nkpt):
        kpt = kpoints[i].frac_coords
        for j in range(ng):
            G_vec = G[j]
            qG = np.round(np.multiply(kpt+G_vec, nk))
            vec_qibzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
            ind_qibzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
    return vec_qibzG_to_ind, ind_qibzG_to_vec

def qGbz_dict(vec_q_to_ind, vec_G_to_ind, nq, ng):
    vec_qbzG_to_ind, ind_qbzG_to_vec = {}, {}
    vec_table = np.zeros((nq*ng, 3))
    for q in vec_q_to_ind.keys():
        i = vec_q_to_ind[q]
        for G_vec in vec_G_to_ind.keys():
            j = vec_G_to_ind[G_vec]
            qG = np.round([q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]])
            vec_qbzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
            ind_qbzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
            vec_table[j+i*ng] = qG
    return vec_qbzG_to_ind, ind_qbzG_to_vec, vec_table

def qG_rot_dict(vec_qibzG_to_ind, vec_qbzG_to_ind, SymRec, nsym):
    vec_qGrot_to_ind, ind_qGrot_to_vec, vec_with_missing_sym = {}, {}, {}
    count = 0
    count_sym = 0
    # Loop over the qibz+G
    for qibzG_vec in vec_qibzG_to_ind.keys():
        #Loop over the symmetry
        for j in range(nsym):
            #Build R(qibz+G)
            qG_rot = np.round(np.matmul(SymRec[j], [qibzG_vec[0], qibzG_vec[1], qibzG_vec[2]]))
            #Check wheter this vector is in the qbz+G set
            if (qG_rot[0], qG_rot[1], qG_rot[2]) not in vec_qbzG_to_ind.keys():
                vec_with_missing_sym[qibzG_vec] = count_sym                    
                count_sym += 1
            #Check wheter it is already in the set (needed so the indices are not counted more than once)
            elif (qG_rot[0], qG_rot[1], qG_rot[2]) in vec_qGrot_to_ind.keys():
                continue
            #If this vector is new and in the qbz+G set, it is added to the list of vectors for which data are available
            else:
                vec_qGrot_to_ind[(qG_rot[0], qG_rot[1], qG_rot[2])] = count
                ind_qGrot_to_vec[count] = (qG_rot[0], qG_rot[1], qG_rot[2])
                count += 1
    return vec_qGrot_to_ind, ind_qGrot_to_vec, vec_with_missing_sym

def smallchi0(smallchi0GG, vec_q_to_ind, vec_G_to_ind, vec_with_missing_sym, nw, ng):
    for q_vec in vec_q_to_ind.keys():
        i = vec_q_to_ind[q_vec]
        for G_vec in vec_G_to_ind.keys():
            j = vec_G_to_ind[G_vec]
            if (q_vec[0]+G_vec[0], q_vec[1]+G_vec[1], q_vec[2]+G_vec[2]) in vec_with_missing_sym.keys():
                smallchi0GG[:, i, j, :] = np.zeros((nw, ng))
                smallchi0GG[:, i, :, j] = np.zeros((nw, ng))
    return smallchi0GG

def build_vec_table_with_border(vec_qbzG_to_ind, ind_qbzG_to_vec, nvec):
    vec_table_with_border = np.zeros((nvec, 3), dtype=int)
    for i in range(nvec):
        qG = ind_qbzG_to_vec[i]
        vec_table_with_border[i] = [qG[0], qG[1], qG[2]]
    return vec_table_with_border

def build_vec_with_all_sym(vec_qibzG_to_ind, vec_with_missing_sym):
    vec_with_all_sym = {}                    
    for qG_vec in vec_qibzG_to_ind.keys():
        if qG_vec not in vec_with_missing_sym.keys():
            vec_with_all_sym[qG_vec] = vec_qibzG_to_ind[qG_vec]
    return vec_with_all_sym

def vec_and_sym(vec_qibzG_to_ind, vec_with_all_sym, nsym, SymRec):
    vec_from_sym, sym_to_vec = {}, {}
    for qG_vec in vec_with_all_sym.keys():
        for j in range(nsym):
            SqG = np.round(np.matmul(SymRec[j],[qG_vec[0], qG_vec[1], qG_vec[2]]))
            if (SqG[0], SqG[1], SqG[2]) in vec_from_sym.keys():
                vec_from_sym[(SqG[0], SqG[1], SqG[2])] = np.append(vec_from_sym[(SqG[0], SqG[1], SqG[2])], j)
            else:
                vec_from_sym[(SqG[0], SqG[1], SqG[2])] = np.array([j])
            sym_to_vec[((SqG[0], SqG[1], SqG[2]), j)] = vec_qibzG_to_ind[qG_vec]
    return vec_from_sym, sym_to_vec


def build_ind_pairqG_to_pair(vec_q_to_ind, vec_G_to_ind, vec_from_sym):
    count = 0
    ind_pairqG_to_pair = {}
    #item_to_loc = {}
    for q_vec in vec_q_to_ind.keys():
        #print(q_vec)
        for G1_vec in vec_G_to_ind.keys():
            qG1_vec = (q_vec[0]+G1_vec[0], q_vec[1]+G1_vec[1], q_vec[2]+G1_vec[2])
            for G2_vec in vec_G_to_ind.keys():
                qG2_vec = (q_vec[0]+G2_vec[0], q_vec[1]+G2_vec[1], q_vec[2]+G2_vec[2])
                if qG1_vec in vec_from_sym.keys() and qG2_vec in vec_from_sym.keys():
                    l1 = vec_from_sym[qG1_vec]
                    l2 = vec_from_sym[qG2_vec]
                    l1_as_set = set(l1)
                    intersection = l1_as_set.intersection(l2)
                    intersection_as_list = list(intersection)
                    ind_pairqG_to_pair[(vec_q_to_ind[q_vec], vec_G_to_ind[G1_vec], vec_G_to_ind[G2_vec])]=(qG1_vec, qG2_vec, intersection_as_list[0])
                    #item_to_loc[count] = 
                    #print(intersection_as_list)
                    count+=1
    return ind_pairqG_to_pair

def Build_Chi0omegaGG_fromSym(filename, nw):
    #Get the data of the input file : the data of the matrix chi0(w, q_ibz, G, G'), the list of qpoints (kpoints), number of G vectors (ng), the number of q-points (nkpt) and the list of G vectors in an array (G)
    sus_ncfile, kpoints, ng, nkpt, G = openfile(filename)
    print(kpoints)
    #Extract the structure from the file
    structure=abipy.core.structure.Structure.from_file(filename)
    dict_struct = structure.to_abivars()
    #Get the primitive vectors
    lattice = structure.lattice.matrix
    A, B, C = lattice[0], lattice[1], lattice[2]
    #Sampling of kpoints grid
    nk = fsk(kpoints)
    vol = np.dot(A, (np.cross(B, C)))*nk[0]*nk[1]*nk[2]
    print("Opening the file" , filename, "containing a matrix chi^0[q, G, G'] with ", nkpt, "q points in the IBZ and ", ng, "G vectors")

    start_time = time.time()

    #extract the data for the symmetries
    Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
    SymRec = Sym.symrec
    nsym = len(SymRec)
    print("The algorithm will use ", nsym, "symmetries to build the matrix chi^0[q, G, G'] with q in the BZ")
    
    #List of the kpoints (scaled)
    qibzvec_to_ind = qibz_dict(kpoints, nk, nkpt)

    #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ. Needs to pay attention to rounding errors+needs to use Umklapp vectors to get all the data (scaled)
    vec_q_to_ind, ind_q_to_vec, sym_dict = qbz_dict(SymRec, nsym, nkpt, kpoints, nk)

    #Verification de l'inclusion de la symétrie d'inversion
    invsym_bool = IsInvSymIn(SymRec, nsym)
    if invsym_bool==False:
        vec_q_to_ind, ind_q_to_vec, sym_dict = add_invsym(vec_q_to_ind, ind_q_to_vec, sym_dict, nsym)
        
    nq = len(sym_dict)
    nqg = nq*ng
    #Liste des vecteurs G (scaled)
    vec_G_to_ind, ind_G_to_vec = G_dict(G, ng, nk)
    
    #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau) (scaled)
    vec_qibzG_to_ind, ind_qibzG_to_vec = qGibz_dict(nkpt, kpoints, ng, G, nk)

    vec_qbzG_to_ind, ind_qbzG_to_vec, vec_table = qGbz_dict(vec_q_to_ind, vec_G_to_ind, nq, ng)
    
    #Liste des vecteurs obtenus par rotation des vecteur de base (qibz+G) + list des vecteurs pour lesquels il n'y pas de reconstruction possible dans la grille définie
    vec_qGrot_to_ind, ind_qGrot_to_vec, vec_with_missing_sym = qG_rot_dict(vec_qibzG_to_ind, vec_qbzG_to_ind, SymRec, nsym)

    nvec = len(vec_qbzG_to_ind) 
    vec_table_with_border = build_vec_table_with_border(vec_qbzG_to_ind, ind_qbzG_to_vec, nvec)
    
    #Settings for the building of chi0GG
    
    s1, s2, s3 = np.amax(np.abs(vec_table_with_border[:, 0])), np.amax(np.abs(vec_table_with_border[:, 1])), np.amax(np.abs(vec_table_with_border[:, 2]))
    n1, n2, n3= (2*s1)+1, (2*s2)+1, (2*s3)+1
    
    chi0GG = np.zeros((nw, nq, ng, ng), dtype = complex)
        
    smallchi0GG = Sym_chi0GG(sus_ncfile, kpoints, ng, nkpt, G, SymRec, nsym, ind_q_to_vec, ind_G_to_vec, qibzvec_to_ind, vec_G_to_ind, vec_qibzG_to_ind, ind_qibzG_to_vec, nk, nw)
    smallchi0GG = smallchi0(smallchi0GG, vec_q_to_ind, vec_G_to_ind, vec_with_missing_sym, nw, ng)
    #print(smallchi0GG)

    
    vec_with_all_sym = build_vec_with_all_sym(vec_qibzG_to_ind, vec_with_missing_sym)

    vec_from_sym, sym_to_vec = vec_and_sym(vec_qibzG_to_ind, vec_with_all_sym, nsym, SymRec)
    #print(vec_from_sym)
    ind_pairqG_to_pair = build_ind_pairqG_to_pair(vec_q_to_ind, vec_G_to_ind, vec_from_sym)
    for omega in range(nw):
        for (i, j, k) in ind_pairqG_to_pair.keys():
            qG1_vec, qG2_vec, S = ind_pairqG_to_pair[(i, j, k)]
            ind_qibzG1, ind_qibzG2 = sym_to_vec[(qG1_vec, S)], sym_to_vec[(qG2_vec, S)]
            ind_sm_chi011,  ind_sm_chi012= ind_qibzG1%ng, ind_qibzG2%ng # Gind
            ind_sm_chi021,  ind_sm_chi022= round((ind_qibzG1-ind_sm_chi011)/ng), round((ind_qibzG2-ind_sm_chi012)/ng) #qind
            if ind_sm_chi021 != ind_sm_chi022:
                print("Strange")
            chi0GG[omega, i, j, k] = smallchi0GG[omega, ind_sm_chi021, ind_sm_chi011,  ind_sm_chi012]#*cmath.exp(ic*np.dot(t, G2_vec-G1_vec))
    chi0GG[:, 0, 0, :] = np.zeros((nw,ng))
    chi0GG[:, 0, :, 0] = np.zeros((nw,ng))
    elapsed = time.time()-start_time
    print("the building of chi0GG took ", elapsed, "seconds")
    return chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol


def Sym_chi0GG(sus_ncfile, kpoints, ng, nkpt, G, SymRec, nsym, ind_q_to_vec, ind_G_to_vec, qibzvec_to_ind, vec_G_to_ind, vec_qibzG_to_ind, ind_qibzG_to_vec, nk, nw):
    #Function that symmetrizes the chi^0GG output of Abinit such that two symmetric values have exactly the same values
    smallchi0GG = np.zeros((nw, nkpt, ng, ng), dtype = complex)
    for i in range(nw):
        for j in range(nkpt):
            q_origin = kpoints[j]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i,j] = chi0[i]

    chi0GGsym = np.zeros((nw, nkpt, ng, ng), dtype = complex)
    #Dictionnary to store all the q, G, G' set already treated
    comb_vec = {}
    count_w = -1
    qGpairs_w = {}
    qGpairs_w[0]=[]
    for i in range(nkpt):
        q = ind_q_to_vec[i]
        qvec = [q[0], q[1], q[2]]
        for G1 in vec_G_to_ind.keys():
            G1vec = [G1[0], G1[1], G1[2]]
            j = vec_G_to_ind[G1]
            for G2 in vec_G_to_ind.keys():
                G2vec = [G2[0], G2[1], G2[2]]
                k = vec_G_to_ind[G2]
                if (i, j, k) not in comb_vec.keys():
                    count_w+=1
                    qGpairs_w[count_w]=[]
                    sum_chi = 0
                    #Dictionnary to store all the symmetrical q,G,G' sets
                    sym_vec = {}
                    count = 0
                    for l in range(nsym):
                        S = SymRec[l]
                        Sq = np.matmul(S, qvec)
                        SG1 = np.matmul(S, G1vec)
                        SG2 = np.matmul(S, G2vec)
                        SqG1 = Sq + SG1
                        SqG2 = Sq + SG2
                        if (SqG1[0], SqG1[1], SqG1[2]) in vec_qibzG_to_ind.keys() and (SqG2[0], SqG2[1], SqG2[2]) in vec_qibzG_to_ind.keys():
                            Sq = Sq/nk
                            if np.amax(q) > 0.5 or np.amin(q) <= - 0.5: #if out of BZ ==> Umklapp
                                a, b, c = Sq[0], Sq[1], Sq[2]
                                if Sq[0]>0.5:
                                    a=round(Sq[0]-1,3)
                                elif Sq[0]<=-0.5:
                                    a=round(Sq[0]+1,3)
                                if Sq[1]>0.5:
                                    b=round(Sq[1]-1,3)
                                elif Sq[1]<=-0.5:
                                    b=round(Sq[1]+1,3)
                                if Sq[2]>0.5:
                                    c=round(Sq[2]-1,3)
                                elif Sq[2]<=-0.5:
                                    c=round(Sq[2]+1,3)
                                Sq_in_bz = (a, b, c)
                                Sq_in_bz = np.round(np.multiply(Sq_in_bz, nk))
                                SG1 = SqG1 - Sq_in_bz
                                SG2 = SqG2 - Sq_in_bz
                            else:
                                Sq_in_bz = (q[0], q[1], q[2])
                                Sq_in_bz = np.round(np.multiply(Sq_in_bz, nk))
                            indq = qibzvec_to_ind[(Sq_in_bz[0], Sq_in_bz[1], Sq_in_bz[2])]
                            indG1 = vec_G_to_ind[(SG1[0], SG1[1], SG1[2])]
                            indG2 = vec_G_to_ind[(SG2[0], SG2[1], SG2[2])]
                            sum_chi += smallchi0GG[0, indq, indG1, indG2]
                            sym_vec[(indq, indG1, indG2)] = l
                            comb_vec[(indq, indG1, indG2)] = 0
                            qGpairs_w[count_w]=np.append(qGpairs_w[count_w], [indq, indG1, indG2])
                            count += 1
                    mean = sum_chi/count
                    #Replacement off all the values in the set by the mean of all the values
                    for (m, n, o) in sym_vec.keys():
                        chi0GGsym[0, m, n, o] = mean               
    for omega in range(1, nw):
        for i in qGpairs_w.keys():
            sum_chi=0
            for j in range(round(len(qGpairs_w[i])/3)):
                ind = j*3
                indq, indG1, indG2 = round(qGpairs_w[i][ind]), round(qGpairs_w[i][ind+1]), round(qGpairs_w[i][ind+2])
                sum_chi+=smallchi0GG[omega, indq, indG1, indG2]
            mean = sum_chi/(round(len(qGpairs_w[i])/3))
            for j in range(round(len(qGpairs_w[i])/3)):
                ind = j*3  
                m,n,o = round(qGpairs_w[i][ind]), round(qGpairs_w[i][ind+1]), round(qGpairs_w[i][ind+2])
                chi0GGsym[omega, m, n, o] = mean           
    return chi0GGsym

def FFT_size(n):
    #0:1, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:8, 8:8, 9:9, 10:10, 11:12, 12:12, 13:15, 14:15, 16:16, 17:18, 18:18, 19:20, 
    #0:1, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:8, 8:8, 9:9, 10:10, 11:12, 12:12, 13:15, 14:15, 16:16, 17:18, 18:18, 19:20, 20:20, 21:24, 22:24, 23:24, 24:24, 25:25, 26:27, 27:27, 28:32, 29:32, 30:32, 31:32, 32:32, 33:36, 34:36, 35:36, 36:36, 37:40, 38:40, 4
    if n >=1000:
        print("this code does not provide the possibility to optimize the size of such a big FFT grid")
        return n
    delta = -1
    i = 0
    values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100, 108, 120, 125, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 225, 240, 243, 250, 256, 270, 288, 300, 320, 324, 360, 375, 384, 400, 405, 432, 450, 480, 486, 500, 512, 540, 576, 600, 625, 640, 648, 675, 720, 729, 750, 768, 800, 810, 864, 900, 960, 972, 1000]
    while delta < 0:
        n_out = values[i]
        delta = n_out - n
        i += 1
    return n_out

def FFT_chi0_from_Mat0(chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol):
    start_time = time.time()
    nq, ng1, ng2 = chi0GG.shape
    if ng1 != ng2:
        return "There is a problem in the code"
    ng=ng1
    nqg = nq * ng
    n1, n2, n3 = round(n1), round(n2), round(n3)
    n1_fft, n2_fft, n3_fft = FFT_size(n1), FFT_size(n2), FFT_size(n3)
    fftboxsize = round(n1_fft*n2_fft*n3_fft)
    maxG1,maxG2,maxG3=np.amax(np.abs(G[:,0])),np.amax(np.abs(G[:,1])),np.amax(np.abs(G[:,2]))
    n4, n5, n6 = int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
    n4_fft, n5_fft, n6_fft = FFT_size(n4), FFT_size(n5), FFT_size(n6)
    #n4_fft, n5_fft, n6_fft = 10, 10, 80
    #print(n4_fft, n5_fft, n6_fft)
    fftboxsizeG = n4_fft*n5_fft*n6_fft
    # Première FFT
    chi0rG = np.zeros((nq, fftboxsizeG, ng), dtype = complex)
    phase_fac = np.ones((n4_fft, n5_fft, n6_fft), dtype = complex)
    ic = complex(0, 1)
    #print(ind_G_to_vec)
    for i in range(nq):
        q = ind_q_to_vec[i]
        #q_vec = np.array([q[0], q[1], q[2]])
        q_vec = np.array([q[0]/nk[0], q[1]/nk[1], q[2]/nk[2]])
        #print(qvec)
        for m in range(n4_fft):
            for n in range(n5_fft):
                for l in range(n6_fft):
                    phase_fac[m, n, l] = cmath.exp(2*math.pi*ic*np.dot(q_vec, [m/n4_fft, n/n5_fft, l/n6_fft])) 
        #print(phase_fac)
        for j in range(ng):
            chi0GqG2 = chi0GG[i, :, j]
            FFTBox = np.zeros((n4_fft, n5_fft, n6_fft), dtype = complex)
            for k in range(ng):
                FFTBox[round(G[k,0]), round(G[k,1]), round(G[k,2])] = chi0GqG2[k]    
            FFT = np.fft.ifftn(FFTBox)
            FFT_out = np.multiply(FFT, phase_fac)
            #add phase factor
            chi0rG[i, :, j]= np.reshape(FFT_out, fftboxsizeG)
    #elapsed1 = time.time()-start_time
    #print(n4_fft, n5_fft, n6_fft )
    #print("The first FFT took ", elapsed1, " seconds")
    # Seconde FFT
    chi0rr = np.zeros((fftboxsizeG, n1_fft, n2_fft, n3_fft), dtype = complex)
    for i in range(fftboxsizeG):
        FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
        for j in range(nq):
            q = ind_q_to_vec[j]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                qG2 = [q[0]+G2[0], q[1]+G2[1], q[2]+G2[2]]
                FFTBox[-round(qG2[0]), -round(qG2[1]), -round(qG2[2])] = chi0rG[j, i, k]
        FFT = np.fft.ifftn(FFTBox)  
        chi0rr[i, :, :, :]=FFT
    #print(n4_fft, n5_fft, n6_fft )
    chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
    chi0rr_out = chi0rr_out0 * chi0rr_out0.size/vol
    elapsed2 = time.time()-start_time
    print("The FFT took ", elapsed2, " seconds")
    return chi0rr_out

def FFT_chi0_from_Mat(chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol):
    start_time = time.time()
    nq, ng1, ng2 = chi0GG.shape
    if ng1 != ng2:
        return "There is a problem in the code"
    ng=ng1
    nqg = nq * ng
    maxG1,maxG2,maxG3=np.amax(np.abs(G[:,0])),np.amax(np.abs(G[:,1])),np.amax(np.abs(G[:,2]))
    n4, n5, n6 = int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
    n4_fft, n5_fft, n6_fft = FFT_size(n4), FFT_size(n5), FFT_size(n6)
    n1_fft, n2_fft, n3_fft = round(n4_fft*nk[0]), round(n5_fft*nk[1]), round(n6_fft*nk[2])
    fftboxsize = round(n1_fft*n2_fft*n3_fft)
    
    #print(n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft)
    #n4_fft, n5_fft, n6_fft = 10, 10, 80
    #print(n4_fft, n5_fft, n6_fft)
    fftboxsizeG = n4_fft*n5_fft*n6_fft
    # Première FFT
    chi0rG = np.zeros((nq, fftboxsizeG, ng), dtype = complex)
    phase_fac = np.ones((n4_fft, n5_fft, n6_fft), dtype = complex)
    ic = complex(0, 1)
    #print(ind_G_to_vec)
    for i in range(nq):
        q = ind_q_to_vec[i]
        #q_vec = np.array([q[0], q[1], q[2]])
        q_vec = np.array([q[0]/nk[0], q[1]/nk[1], q[2]/nk[2]])
        #print(qvec)
        for m in range(n4_fft):
            for n in range(n5_fft):
                for l in range(n6_fft):
                    phase_fac[m, n, l] = cmath.exp(2*math.pi*ic*np.dot(q_vec, [m/n4_fft, n/n5_fft, l/n6_fft])) 
        #print(phase_fac)
        for j in range(ng):
            chi0GqG2 = chi0GG[i, :, j]
            FFTBox = np.zeros((n4_fft, n5_fft, n6_fft), dtype = complex)
            for k in range(ng):
                FFTBox[round(G[k,0]), round(G[k,1]), round(G[k,2])] = chi0GqG2[k]    
            FFT = np.fft.ifftn(FFTBox)
            FFT_out = np.multiply(FFT, phase_fac)
            #add phase factor
            chi0rG[i, :, j]= np.reshape(FFT_out, fftboxsizeG)
    #elapsed1 = time.time()-start_time
    #print(n4_fft, n5_fft, n6_fft )
    #print("The first FFT took ", elapsed1, " seconds")
    # Seconde FFT
    chi0rr = np.zeros((fftboxsizeG, n1_fft, n2_fft, n3_fft), dtype = complex)
    for i in range(fftboxsizeG):
        FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
        for j in range(nq):
            q = ind_q_to_vec[j]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                qG2 = [q[0]+G2[0], q[1]+G2[1], q[2]+G2[2]]
                FFTBox[-round(qG2[0]), -round(qG2[1]), -round(qG2[2])] = chi0rG[j, i, k]
        FFT = np.fft.ifftn(FFTBox)  
        chi0rr[i, :, :, :]=FFT
    #print(n4_fft, n5_fft, n6_fft )
    chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
    chi0rr_out = chi0rr_out0 * chi0rr_out0.size/vol
    elapsed2 = time.time()-start_time
    print("The FFT took ", elapsed2, " seconds")
    return chi0rr_out

def FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, n1, n2, n3, n4, n5, n6, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2 = "Kaltak"):
    start_time = time.time()
    nq, ng1, ng2 = chi0GG.shape
    if ng1 != ng2:
        return "There is a problem in the code"
    ng=ng1
    nqg = nq * ng
    n1, n2, n3 = round(n1), round(n2), round(n3)
    n1_fft, n2_fft, n3_fft = n1, n2, n3
    fftboxsize = round(n1_fft*n2_fft*n3_fft)
    n4_fft, n5_fft, n6_fft = n4, n5, n6
    print(n4_fft, n5_fft, n6_fft)
    fftboxsizeG = n4_fft*n5_fft*n6_fft
    # Première FFT
    chi0rG = np.zeros((nq, fftboxsizeG, ng), dtype = complex)
    phase_fac = np.ones((n4_fft, n5_fft, n6_fft), dtype = complex)
    ic = complex(0, 1)
    #print(ind_G_to_vec)
    for i in range(nq):
        q = ind_q_to_vec[i]
        #q_vec = np.array([q[0], q[1], q[2]])
        q_vec = np.array([q[0]/nk[0], q[1]/nk[1], q[2]/nk[2]])
        #print(qvec)
        for m in range(n4_fft):
            for n in range(n5_fft):
                for l in range(n6_fft):
                    phase_fac[m, n, l] = cmath.exp(2*math.pi*ic*np.dot(q_vec, [m/n4_fft, n/n5_fft, l/n6_fft])) 
        #print(phase_fac)
        for j in range(ng):
            chi0GqG2 = chi0GG[i, :, j]
            FFTBox = np.zeros((n4_fft, n5_fft, n6_fft), dtype = complex)
            for k in range(ng):
                FFTBox[round(G[k,0]), round(G[k,1]), round(G[k,2])] = chi0GqG2[k]    
            FFT = np.fft.ifftn(FFTBox)
            FFT_out = np.multiply(FFT, phase_fac)
            #add phase factor
            chi0rG[i, :, j]= np.reshape(FFT_out, fftboxsizeG)
    #print(n4_fft, n5_fft, n6_fft )
    #print("The first FFT took ", elapsed1, " seconds")
    # Seconde FFT
    chi0rr = np.zeros((fftboxsizeG, n1_fft, n2_fft, n3_fft), dtype = complex)
    for i in range(fftboxsizeG):
        #print(i/(fftboxsizeG-1))
        FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
        for j in range(nq):
            q = ind_q_to_vec[j]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                qG2 = [q[0]+G2[0], q[1]+G2[1], q[2]+G2[2]]
                FFTBox[-round(qG2[0]), -round(qG2[1]), -round(qG2[2])] = chi0rG[j, i, k]
        FFT = np.fft.ifftn(FFTBox)  
        chi0rr[i, :, :, :]=FFT
    chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
    chi0rr_out = chi0rr_out0 * chi0rr_out0.size/vol
    elapsed2 = time.time()-start_time
    print("The FFT took ", elapsed2, " seconds")
    return chi0rr_out
    