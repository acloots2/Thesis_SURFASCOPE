######Xavier's Algorithm#########
from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
import abipy
from abipy.electrons.scr import ScrFile
import numpy as np
import cmath
import math
import pointcloud as pc
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
import time

def chi0wzz(filenameS, filenameB, nc, opt = "FromSym", opt2 = "Kaltak", axe = "001"):
    sus_ncfileS = ScrFile(filenameS)
    sus_ncfileB = ScrFile(filenameB)
    nwS = sus_ncfileS.reader.nw
    nwB = sus_ncfileB.reader.nw
    print(nwS)
    print(nwB)
    if nwS != nwB:
        print("nomega must be the same for the slab and the bulk. Please provide files with the same frequency sampling")
        return 0
    else:
        for i in range(nwS):
            chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0GG(filenameS, opt, i)
            chi0rr = FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, n1*nc[0], n2*nc[1], n3, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2)
            chi0zzS = chi0zz_test(chi0rr, axe)
            
            chi0GG, ind_qG_to_vec, d1, d2, d3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0GG(filenameB, opt, i)
            chi0rr = FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, d1*nc[0], d2*nc[1], d3*nc[2], d1, d2, d3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2)
            chi0zzB = chi0zz_test(chi0rr, axe)
            
            Lchi0zz = LargeChi0(chi0zzB, chi0zzS)
            if i==0:
                l1, l2 = Lchi0zz.shape
                chi0wzz = np.zeros(nwS, l1, l2, dtype = complex)
            chi0wzz[i, :, :] = Lchi0zz
        return chi0wzz

def chi0wzz_test(filenameS, filenameB, nc, opt = "FromSym", opt2 = "Kaltak", axe = "001"):
    sus_ncfileS = ScrFile(filenameS)
    sus_ncfileB = ScrFile(filenameB)
    nwS = sus_ncfileS.reader.nw
    nwB = sus_ncfileB.reader.nw
    print(nwS)
    print(nwB)
    if nwS != nwB:
        print("nomega must be the same for the slab and the bulk. Please provide files with the same frequency sampling")
        return 0
    else:
        for i in range(nwS):
            chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0GG(filenameS, opt, i)
            chi0rr = FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, n1*nc[0], n2*nc[1], n3, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2)
            chi0zzS = chi0zz_test(chi0rr, axe)
            
            chi0GG, ind_qG_to_vec, d1, d2, d3, ind_q_to_vec, ind_G_to_vec, G, nk, vol = Build_Chi0GG(filenameB, opt, i)
            chi0rr = FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, d1*nc[0], d2*nc[1], d3*nc[2], d1, d2, d3, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2)
            chi0zzB = chi0zz_test(chi0rr, axe)
            
            Lchi0zz = LargeChi0(chi0zzB, chi0zzS)
            if i==0:
                l1, l2 = Lchi0zz.shape
                chi0wzz = np.zeros(nwS, l1, l2, dtype = complex)
            chi0wzz[i, :, :] = Lchi0zz
        return chi0wzz    

def chi0zz_test(chi0rr, axe = "001"):
    d1, d2, d3, d4, d5, d6 = chi0rr.shape
    print(d3, d6)
    if axe == "001":
        print("in")
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
        chi0zz= chi0zz/(d1*d3)**2
    if axe == "100":
        chi0zz = np.zeros((d1, d4), dtype = complex)
        for i in range(d1):
            for j in range(d4):
                chi0zz[i, j]=np.sum(chi0rr[i, :, :, j, :, :])
        chi0zz= chi0zz/(d2*d3)**2
    return chi0zz

def chi0zz(chi0rr, axe = "001"):
    d1, d2, d3, d4, d5, d6 = chi0rr.shape
    if axe == "001":
        chi0zz = np.zeros((d3, d6), dtype = complex)
        for i in range(d1):
            for j in range(d2):
                chi0zz=chi0zz+chi0Comp(chi0rr, i, j, axe = "001")
        chi0zz= chi0zz/(d1*d2)
    if axe == "010":
        chi0zz = np.zeros((d2, d5), dtype = complex)
        for i in range(d1):
            for j in range(d3):
                chi0zz=chi0zz+chi0Comp(chi0rr, i, j, axe = "010")
        chi0zz= chi0zz/(d1*d3)
    if axe == "100":
        chi0zz = np.zeros((d1, d4), dtype = complex)
        for i in range(d2):
            for j in range(d3):
                chi0zz=chi0zz+chi0Comp(chi0rr, i, j, axe = "100")
        chi0zz= chi0zz/(d2*d3)
    return chi0zz

#Build the matrix chi0(z, z') from chi0(r, r')
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
                arr = Rev_vec(chi0zz[i-count-2, :])
                count+=2
                arr = arr[::-1]
                chi0zz_out[i, :] = arr
    for j in range(d2):
        chi0zz_out[:, j] = Rev_vec(chi0zz_out[:, j])
    return chi0zz_out



        
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
    
#r'$\chi^0(\boldymbol{r}, \boldsymbol{r\'})$' + 'with' + r'$\boldymbol{r}$' +'='+str([0, 0, 0])+' for different directions'
def getProfile(chi0rr, R, a, b, c, nc, axe = "100"):
    fig = go.Figure()
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    Rreal = np.round(np.multiply(np.array(R)/np.array([n1, n2, n3]), [a, b, c]), 3)
    if axe == "100":
        if n4%2==0:
            endpoint = False
        else:
            endpoint = True
        X = np.linspace(-a/2, a/2, n4,endpoint = endpoint)
        Y = Rev_vec(np.real(chi0rr[R[0], R[1], R[2], :, R[1], R[2]]))
    elif axe == "010":
        if n5%2==0:
            endpoint = False
        else:
            endpoint = True
        X = np.linspace(-b/2, b/2, n5,endpoint = endpoint)
        Y = Rev_vec(np.real(chi0rr[R[0], R[1], R[2], R[0], :, R[2]]))
    elif axe == "001":
        if n6%2==0:
            endpoint = False
        else:
            endpoint = True
        X = np.linspace(-c/2, c/2, n6, endpoint = endpoint)
        Y = Rev_vec(np.real(chi0rr[R[0], R[1], R[2], R[0], R[1], :]))
    else:
        print("This possibility is not available yet, please select among the following : 100, 010 or 001")
    xtitle = r'$\boldsymbol{r}_2 \text{ on the axis parallel to }['+axe+ r'] \text{ and passing by the point }['+str(Rreal[0])+r', '+str(Rreal[1])+r', '+str(Rreal[2])+ r'] \text{\AA} $'
    fig.add_trace(go.Scatter(x = X, y = np.real(Y), mode = "lines+markers",marker=dict(
            color='slategrey',
            size=10,
        ),))
    fig.update_layout(title_text = r'$\chi^0(\boldsymbol{r}_1, \boldsymbol{r}_2) \text{ with } \boldsymbol{r}_1 = ['+str(Rreal[0])+r', '+str(Rreal[1])+r', '+str(Rreal[2])+ r'] \text{ angstrom}$', title_x=0.5,xaxis_title= xtitle,
             yaxis_title = r'$\chi^0(\boldsymbol{r}_1, \boldsymbol{r}_2)$',paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',)
    fig.show(dpi = 300)

def diff_bulk_Slab(chi0rrB, chi0rrS, nc, dr, cut = 0, plan = "xz"):
    charge_cut_b = charge_accul_cut(chi0rrB, dr, plan, cut)
    charge_cut_s = charge_accul_cut(chi0rrS, dr, plan, cut)
    nb1, nb2 = charge_cut_b.shape
    ns1, ns2 = charge_cut_s.shape
    nc1, nc2 = round(ns1/nb1), round(ns2/nb2)
    charge_cut_b = charge_cut_b[0:nb1-1, 0:nb2-1]
    nb1, nb2 = charge_cut_b.shape
    charge_cut_diff = np.zeros((ns1, ns2), dtype = complex)
    for i in range(nc1):
        for j in range(nc2):
            charge_cut_diff[i*nb1:(i+1)*nb1, j*nb2:(j+1)*nb2] = charge_cut_b
    charge_cut_diff = charge_cut_s-charge_cut_diff
    return charge_cut_diff
        


def getDen(filename, nx, nz):
    dataDEN = np.genfromtxt(filename)
    z = dataDEN[:, 0]
    x = dataDEN[:, 1]
    den = dataDEN[:, 2]

    Z0 = z[0]
    Z1 = z[len(x)-1]
    X0 = x[0]
    X1 = x[len(z)-1]
    print(X0, X1, Z0, Z1)
    Y = np.linspace(Z0, Z1, nz)
    X = np.linspace(X0, X1, nx)
    den = np.reshape(den, (nz, nx))

    R = nx/nz
    fig = go.Figure(data = go.Contour(z = den, x = X, y = Y, colorscale='sunset', line_smoothing=0.85, contours_coloring='heatmap', ncontours = 8))
    if R==1:
        fig.update_layout(
            autosize=False,
            width=800,
            height=800,
            )
    elif R>1:
        fig.update_layout(
            autosize=False,
            width=500*R,
            height=500,
            )
    else :
        fig.update_layout(
            autosize=False,
            width=500,
            height=500*R,
            )

    fig.show(dpi = 300)

def Rev_vec(Y):
    l1 = Y.size
    Y_out = np.zeros((l1), dtype = complex)
    if l1%2==0:
        l2 = math.floor(l1/2)
        Y_out[0:l2] = Y[l2:l1]
        Y_out[l2:l1] = Y[0:l2]
    else: 
        l2 = math.floor(l1/2)
        Y_out[0:l2] = Y[l2+1:l1]
        Y_out[l2:l1] = Y[0:l2+1]
    return Y_out

def charge_nearby(chi0rr, dV, dr):
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    charge = 0

    for i in range(dV[0]-dr, dV[0]+dr+1):
        if i >= n4:
            i = i - n4
        for j in range(dV[1]-dr, dV[1]+dr+1):
            if j >= n5:
                j = j - n5
            for k in range(dV[2]-dr, dV[2]+dr+1):
                if k >= n6:
                    k = k - n6
                charge += chi0rr[dV[0], dV[1], dV[2], i, j, k]
    
    return charge

def charge_evol(chi0rr, dr):
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    charge_evol = np.zeros((n1, n2, n3), dtype = complex)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                dV = [i, j, k]
                charge_evol[i, j, k] = charge_nearby(chi0rr, dV, dr)
    return charge_evol

def charge_accul_cut(chi0rr, dr, plan, cut):
    charge = charge_evol(chi0rr, dr)
    d1, d2, d3 = charge.shape
    if plan == "xy":
        charge_cut = np.zeros((d1+1, d2+1), dtype = complex)
        charge_cut[0:d1, 0:d2] = charge[:, :, cut]
        charge_cut[d1, :] = charge_cut[0, :]
        charge_cut[:, d2] = charge_cut[:, 0]  
    elif plan == "xz":
        charge_cut = np.zeros((d1+1, d3+1), dtype = complex)
        charge_cut[0:d1, 0:d3] = charge[:, cut, :]
        charge_cut[d1, :] = charge_cut[0, :]
        charge_cut[:, d3] = charge_cut[:, 0]
    elif plan == "yz":
        charge_cut = np.zeros((d2+1, d3+1), dtype = complex)
        charge_cut[0:d2, 0:d3] = charge[cut, :, :]
        charge_cut[d2, :] = charge_cut[0, :]
        charge_cut[:, d3] = charge_cut[:, 0]
    print("Max charge accumulation = ", np.amax(np.abs(np.real(charge_cut))))
    return charge_cut

def getChargeInBox_centered(charge_cut_raw, a, b, c, opt = "xz"):
    l1, l2 = charge_cut_raw.shape
    charge_cut_inter = np.zeros((l1-1, l2-1), dtype = complex)
    for i in range(l1-1):
        charge_cut_inter[i]=Rev_vec(charge_cut_raw[i, 0:l2-1])
    for j in range(l2-1):
        charge_cut_inter[:, j] = Rev_vec(charge_cut_inter[0:l1-1, j])
    if l1%2==0 and l2%2==0:
        charge_cut = np.zeros((l1-1, l2-1), dtype = complex)
        charge_cut = charge_cut_inter
        l1-=l1
        l2-=l2
    elif l1%2==1 and l2%2==0:
        charge_cut = np.zeros((l1, l2-1), dtype = complex)
        charge_cut[0:l1-1, 0:l2-1] = charge_cut_inter
        charge_cut[l1-1, :] = charge_cut[0, :]
        l2-=l2
    elif l1%2==0 and l2%2==1:
        charge_cut = np.zeros((l1-1, l2), dtype = complex)
        charge_cut[0:l1-1, 0:l2-1] = charge_cut_inter
        charge_cut[:, l2-1] = charge_cut[:, 0]
        l1-=l1
    else:
        charge_cut = np.zeros((l1, l2), dtype = complex)
        charge_cut[0:l1-1, 0:l2-1] = charge_cut_inter
        charge_cut[l1-1, :] = charge_cut[0, :]
        charge_cut[:, l2-1] = charge_cut[:, 0]
    if opt == "xz":
        s1 = c
        s2 = a
        xaxis_title="z"
        yaxis_title="x"
    elif opt == "xy":
        s1 = b
        s2 = a
        xaxis_title="y"
        yaxis_title="x"
    elif opt == "yz":
        s1 = c
        s2 = b
        xaxis_title="z"
        yaxis_title="y"
    else:
        print("Not a valid option, please select one among the following : xz, xy, yz")
    fig = go.Figure(data = go.Contour(z = np.real(charge_cut), x = np.linspace(-s1/2, s1/2, l2), y = np.linspace(-s2/2, s2/2, l1), colorscale='sunset', ncontours = 20,  line_smoothing=0.85, contours_coloring='heatmap', colorbar=dict(title='Amount of charge',)))
    fig.update_layout(
        title_text="Sum of the charge in a box center around the point shown in the "+opt+" plane of the unit cell",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        autosize=False,
        width=600*c/a,
        height=600,
        )
    fig.show(dpi = 300)

def getChargeInBox(charge_cut, a, b, c, opt = "xz"):
    l1, l2 = charge_cut.shape
    if opt == "xz":
        s1 = c
        s2 = a
    elif opt == "xy":
        s1 = b
        s2 = a
    elif opt == "yz":
        s1 = c
        s2 = b
    else:
        print("Not a valid option, please select one among the following : xz, xy, yz")
    fig = go.Figure(data = go.Contour(z = np.real(charge_cut), x = np.linspace(-s1/2, s1/2, l2), y = np.linspace(-s2/2, s2/2, l1), colorscale='sunset', line_smoothing=0.85, contours_coloring='heatmap'))
    fig.update_layout(
        autosize=False,
        width=600*c/a,
        height=600,
        )
    fig.show(dpi = 300)

def getChi0cut(chi0rr, R, a, b, c, nc, cut = 0,  opt = "xz"):
    l1, l2, l3 = chi0rr[R[0], R[1], R[2], :, :, :].shape
    if opt == "xz":
        s1 = c*nc[2]
        s2 = a*nc[0]
        d1 = l3
        d2 = l1
        chi_cut = chi0rr[R[0], R[1], R[2], :, cut, :]
        print(chi_cut.shape)
        print(d1, d2)
    elif opt == "xy":
        s1 = b*nc[1]
        s2 = a*nc[0]
        d1 = l2
        d2 = l1
        chi_cut = chi0rr[R[0], R[1], R[2], :, :, cut]
    elif opt == "yz":
        s1 = c*nc[2]
        s2 = b*nc[1]
        d1 = l3
        d2 = l2
        chi_cut = chi0rr[R[0], R[1], R[2], cut, :, :]
    else:
        print("Not a valid option, please select one among the following : xz, xy, yz")
    fig = go.Figure(data = go.Contour(z = np.real(chi_cut), x = np.linspace(-s1/2, s1/2, d1), y = np.linspace(-s2/2, s2/2, d2), colorscale='sunset', line_smoothing=0.85, contours_coloring='heatmap'))
    fig.update_layout(
        autosize=False,
        width=600*s1/s2,
        height=600,
        )
    fig.show(dpi = 300)

def Sym_chi0GG(sus_ncfile, kpoints, ng, nkpt, G, SymRec, nsym, ind_q_to_vec, ind_G_to_vec, qibzvec_to_ind, vec_G_to_ind, vec_qibzG_to_ind, ind_qibzG_to_vec, nk, omega):
    # Function that symmetrizes the chi^0GG output of Abinit such that two symmetric values have exactly the same values
    smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)
    for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega]

    chi0GGsym = np.zeros((nkpt, ng, ng), dtype = complex)
    #Dictionnary to store all the q, G, G' set already treated
    comb_vec = {}
    for i in range(nkpt):
        q = ind_q_to_vec[i]
        qvec = [q[0], q[1], q[2]]
        for j in range(ng):
            G1 = ind_G_to_vec[j]
            G1vec = [G1[0], G1[1], G1[2]]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                G2vec = [G2[0], G2[1], G2[2]]
                if (i, j, k) not in comb_vec.keys():
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
                            sum_chi += smallchi0GG[indq, indG1, indG2]
                            sym_vec[(indq, indG1, indG2)] = l
                            comb_vec[(indq, indG1, indG2)] = 0
                            count += 1
                    mean = sum_chi/count
                    #Replacement off all the values in the set by the mean of all the values
                    for (m, n, o) in sym_vec.keys():
                        chi0GGsym[m, n, o] = mean
    return chi0GGsym


def Build_Chi0GG(filename, opt, omega = 0):
    sus_ncfile, kpoints, ng, nkpt, G = openfile(filename)
    structure=abipy.core.structure.Structure.from_file(filename)
    dict_struct = structure.to_abivars()
    lattice = structure.lattice.matrix
    A, B, C = lattice[0], lattice[1], lattice[2]
    nk = fsk(kpoints)
    vol = np.dot(A, (np.cross(B, C)))*nk[0]*nk[1]*nk[2]
    print("Opening the file" , filename, "containing a matrix chi^0[q, G, G'] with ", nkpt, "q points in the IBZ and ", ng, "G vectors")
    if opt == 'FullBZ':
        nvec = nkpt*ng
        print(nkpt, ng, nvec)
        #Creation des dictionnaires : il en faut 1 pour aller des indices vers les vecteurs et un pour aller des vecteurs vers les indices. Les composant sont scale selon le nombre 
        # le sampling de la BZ (ex : le vecteur [c0, c1, c2] est référencé à (c0*ngkpt[0], c1*ngkpt[1], c2*ngkpt[2])) de manière à ce qu'ils puissent être utilisé comme indices dans les matrices.
        vec_qG_to_ind = {}
        ind_qG_to_vec = {}
        ind_q_to_vec = {}
        vec_q_to_ind = {}
        ind_G_to_vec = {}
        vec_G_to_ind = {}
        
        #Initialisation d'un tableau pour garder les vecteurs q+G et pouvoir récupérer les données plus facilement que dans un dict.

        vec_table = np.zeros((nvec,3), dtype = int)

        #Mise en mémoire de tous les vecteurs (q, G et q+G)
        for i in range(nkpt):
            q = kpoints[i].frac_coords
            q_vec = np.round(np.multiply([q[0], q[1], q[2]], nk))
            ind_q_to_vec[i] = q_vec
            vec_q_to_ind[(q_vec[0], q_vec[1], q_vec[2])] = i
            for j in range(ng):
                if i == 0:
                    G_vec = G[j]
                    ind_G_to_vec[j] = np.round(np.multiply([G_vec[0], G_vec[1], G_vec[2]], nk))
                    vec_G_to_ind[(round(G_vec[0]*nk[0]), round(G_vec[1]*nk[1]), round(G_vec[2]*nk[2]))] = j
                ind = j + i * ng
                vec_table[ind] = np.round(np.multiply((kpoints[i].frac_coords + G[j]), nk))
                qG = (vec_table[ind,0], vec_table[ind,1], vec_table[ind,2])
                vec_qG_to_ind[qG] = ind 
                ind_qG_to_vec[ind] = qG
        
        #print(ind_q_to_vec)
        #Creation d'un second dictionnaire sans les points frontières :

        vec_qG_to_ind_without_border = {}
        ind_qG_to_vec_without_border = {}
        count = 0
        for i in range(nvec):
            qG = ind_qG_to_vec[i]
            qGopp = (-qG[0], -qG[1], -qG[2])
            if qGopp not in vec_qG_to_ind.keys():
                #print('q+G=[',qG,'] of norm ',np.sum(np.power([qG[0],qG[1],qG[2]],2)),' has no opposite in the set available')
                continue
            else:
                vec_qG_to_ind_without_border[qG] = count
                ind_qG_to_vec_without_border[count] = qG
                count += 1
        #print(count)
        """ vec_table_without_border = np.zeros((count, 3), dtype = int)
        for i in range(count):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]""" 

        s1, s2, s3 = np.amax(np.abs(vec_table[:, 0])), np.amax(np.abs(vec_table[:,1])), np.amax(np.abs(vec_table[:,2]))
        n1, n2, n3 = 2*s1+1, 2*s2+1, 2*s3+1
        #print(n1, n2, n3) 
        """ s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:,1])), np.amax(np.abs(vec_table_without_border[:,2]))
        n1, n2, n3 = 2*s1+1, 2*s2+1, 2*s3+1 """
        
        #Initialisation of chi0 :

        chi0GG = np.zeros((nkpt, ng, ng), dtype = complex)
        for i in range(nkpt):
            chi0 = sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat
            chi0GG[i, :, :] = chi0[omega]
            for j in range(ng):
                qG = ind_qG_to_vec[j+i*ng]
                if qG not in vec_qG_to_ind_without_border.keys():
                    chi0GG[i, j, :] = np.zeros(ng)
                    chi0GG[i, :, j] = np.zeros(ng)
        #print(chi0GG)   
        

        return chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk
        
    elif opt == "FromSym":
        start_time = time.time()
        structure = abipy.core.structure.Structure.from_file(filename)
        Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
        dict_struct = structure.to_abivars()
        SymRec = Sym.symrec
        nsym = len(SymRec)
        print("The algorithm will use ", nsym, "symmetries to build the matrix chi^0[q, G, G'] with q in the BZ")

        #Value used to scale all the points in order to have only integers
        nk = fsk(kpoints)

        qibzvec_to_ind  = {}
        for i in range(nkpt):
            q = np.round(np.multiply(kpoints[i].frac_coords, nk))
            qibzvec_to_ind[(q[0], q[1], q[2])] = i

        #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ. Needs to pay attention to rounding errors+needs to use Umklapp vectors to get all the data
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

        #Verification de l'inclusion de la symétrie d'inversion
        invsym_bool = IsInvSymIn(SymRec, nsym)
        if invsym_bool==False:
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
        nq = len(sym_dict)

        #Liste des vecteurs G
        vec_G_to_ind = {}
        ind_G_to_vec = {}
        for i in range(ng):
            G_vec = np.round(np.multiply([G[i, 0], G[i, 1], G[i, 2]], nk))
            vec_G_to_ind[(G_vec[0], G_vec[1], G_vec[2])] = i
            ind_G_to_vec[i] = (G_vec[0], G_vec[1], G_vec[2])
    
        #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau)
        vec_qibzG_to_ind = {}
        ind_qibzG_to_vec = {}
        vec_qbzG_to_ind = {}
        ind_qbzG_to_vec ={}

        for i in range(nkpt):
            kpt = kpoints[i].frac_coords
            for j in range(ng):
                G_vec = G[j]
                qG = np.round(np.multiply(kpt+G_vec, nk))
                vec_qibzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qibzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
        
        nqibzG = nkpt*ng
        vec_table = np.zeros((nq*ng, 3))
        for i in range(nq):
            q = ind_q_to_vec[i]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                qG = np.round([q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]])
                vec_qbzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qbzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
                vec_table[j+i*ng] = qG
        
        
        vec_qGrot_to_ind = {}
        ind_qGrot_to_vec = {}

        
        #Builds all the vectors possible from symmetry. If one of them is not in the qbz+G set, the qibz+G vector is retained in a list 
        # in order to put all the values chi^0_qibz(G, G') and chi^0_qibz(G', G) to 0 for all G'
     
        vec_with_missing_sym = {}
        count = 0
        count_sym = 0
        # Loop over the qibz+G
        for i in range(nqibzG):
            qibzG_vec = ind_qibzG_to_vec[i]
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

        
        nvec = len(vec_qbzG_to_ind) 
        vec_table_with_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qbzG_to_vec[i]
            vec_table_with_border[i] = [qG[0], qG[1], qG[2]]
        

        #Settings for the building of chi0GG
               
        #print("Dict of symmetry initialized")

        s1, s2, s3 = np.amax(np.abs(vec_table_with_border[:, 0])), np.amax(np.abs(vec_table_with_border[:, 1])), np.amax(np.abs(vec_table_with_border[:, 2]))
        n1, n2, n3= (2*s1)+1, (2*s2)+1, (2*s3)+1
        elapsed_1 = time.time()-start_time
        print("the initialization of the dictionnaries and gathering of the information about the valid vectors took", elapsed_1, "seconds")
        chi0GG = np.zeros((nq, ng, ng), dtype = complex)
        #smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)

        """ for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega] """
        smallchi0GG = Sym_chi0GG(sus_ncfile, kpoints, ng, nkpt, G, SymRec, nsym, ind_q_to_vec, ind_G_to_vec, qibzvec_to_ind, vec_G_to_ind, vec_qibzG_to_ind, ind_qibzG_to_vec, nk, omega)
        #smallchi0GG = Sym_chi0GG(filename, omega)

        for i in range(nkpt):
            q_vec = ind_q_to_vec[i]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                if (q_vec[0]+G_vec[0], q_vec[1]+G_vec[1], q_vec[2]+G_vec[2]) in vec_with_missing_sym.keys():
                    smallchi0GG[i, j, :] = np.zeros(ng)
                    smallchi0GG[i, :, j] = np.zeros(ng)
        #print(smallchi0GG)
        elapsed_2 = time.time()-(elapsed_1+start_time)
        print("The initialisation of smallchi0GG took", elapsed_2, "seconds")
        
        vec_with_all_sym = {}                    
        for qG_vec in vec_qibzG_to_ind.keys():
            if qG_vec not in vec_with_missing_sym.keys():
                vec_with_all_sym[qG_vec] = vec_qibzG_to_ind[qG_vec]
        vec_from_sym = {}
        sym_to_vec = {}
        for qG_vec in vec_with_all_sym.keys():
            for j in range(nsym):
                SqG = np.round(np.matmul(SymRec[j],[qG_vec[0], qG_vec[1], qG_vec[2]]))
                if (SqG[0], SqG[1], SqG[2]) in vec_from_sym.keys():
                    vec_from_sym[(SqG[0], SqG[1], SqG[2])] = np.append(vec_from_sym[(SqG[0], SqG[1], SqG[2])], j)
                else:
                    vec_from_sym[(SqG[0], SqG[1], SqG[2])] = np.array([j])
                sym_to_vec[((SqG[0], SqG[1], SqG[2]), j)] = vec_qibzG_to_ind[qG_vec]
        #print(vec_from_sym)
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
        """ ic = complex(0, 1)
            t = Trec[S]
            G1 = ind_G_to_vec[j]
            G1_vec = np.array([G1[0], G1[1], G1[2]])
            G2 = ind_G_to_vec[k]
            G2_vec = np.array([G2[0], G2[1], G2[2]]) """
        for (i, j, k) in ind_pairqG_to_pair.keys():
            qG1_vec, qG2_vec, S = ind_pairqG_to_pair[(i, j, k)]
            ind_qibzG1, ind_qibzG2 = sym_to_vec[(qG1_vec, S)], sym_to_vec[(qG2_vec, S)]
            ind_sm_chi011,  ind_sm_chi012= ind_qibzG1%ng, ind_qibzG2%ng # Gind
            ind_sm_chi021,  ind_sm_chi022= round((ind_qibzG1-ind_sm_chi011)/ng), round((ind_qibzG2-ind_sm_chi012)/ng) #qind
            if ind_sm_chi021 != ind_sm_chi022:
                print("Strange")
            chi0GG[i, j, k] = smallchi0GG[ind_sm_chi021, ind_sm_chi011,  ind_sm_chi012]#*cmath.exp(ic*np.dot(t, G2_vec-G1_vec))
        print(np.amax(np.abs(np.real(chi0GG[0, 0, :]))))
        print(np.amax(np.abs(np.real(chi0GG[0, :, 0]))))
        print(np.amax(np.abs(np.imag(chi0GG[0, 0, :]))))
        print(np.amax(np.abs(np.imag(chi0GG[0, :, 0]))))
        chi0GG[0, 0, :] = np.zeros(ng)
        chi0GG[0, :, 0] = np.zeros(ng)
        elapsed_3 = time.time()-(elapsed_2+elapsed_1+start_time)
        print("the building of chi0GG took ", elapsed_3, "seconds")
        return chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, vol
    
    else:
        return str(opt)+' is not a valid option, read the documentation to see the list of options'

def FFT_chi0(filename, opt1 = "FromSym", opt2 = "Kaltak", omega = 0):
    if opt2 == "Standard":
        chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk = Build_Chi0GG(filename, opt1, omega)
        nq, ng1, ng2 = chi0GG.shape
        if ng1 != ng2:
            print("There is a problem in the code")
            return "There is a problem in the code"
        ng=ng1
        nqg = ng*nq
        n1, n2, n3 = round(n1), round(n2), round(n3)
        n1_fft, n2_fft, n3_fft = FFT_size(n1), FFT_size(n2), FFT_size(n3)
        fftboxsize = round(n1_fft*n2_fft*n3_fft)
        chi0rG = np.zeros((fftboxsize, nqg), dtype = complex)
        print("Starting first FFT")
        for i in range(nq):
            q_vec = ind_q_to_vec[i]
            q = [q_vec[0], q_vec[1], q_vec[2]]
            for j in range(ng): 
                FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)               
                for k in range(ng):
                    G_vec = ind_G_to_vec[k]
                    qG_vec_fft1 = [q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]]
                    FFTBox[round(qG_vec_fft1[0]), round(qG_vec_fft1[1]), round(qG_vec_fft1[2])] = chi0GG[i, j, k]
                FFT = np.fft.ifftn(FFTBox)
                chi0rG[:, j+i*ng] = np.reshape(FFT, fftboxsize)


        print("Starting second FFT")
        chi0rr = np.zeros((fftboxsize, n1_fft, n2_fft, n3_fft), dtype = complex)
        for i in range(fftboxsize):
            FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
            for j in range(nqg):
                qG_vec_fft2 = ind_qG_to_vec[j]
                FFTBox[-round(qG_vec_fft2[0]), -round(qG_vec_fft2[1]), -round(qG_vec_fft2[2])] = chi0rG[i, j]
            FFT = np.fft.ifftn(FFTBox) 
            chi0rr[i, :, :, :]=FFT
        chi0rr_out0 = np.reshape(chi0rr, (n1_fft, n2_fft, n3_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        return chi0rr_out

    elif opt2 == "Kaltak":
        chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk = Build_Chi0GG(filename, opt1, omega)
        nq, ng1, ng2 = chi0GG.shape
        if ng1 != ng2:
            return "There is a problem in the code"
        ng=ng1
        nqg = nq * ng
        n1, n2, n3 = round(n1), round(n2), round(n3)
        n1_fft, n2_fft, n3_fft = FFT_size(n1), FFT_size(n2), FFT_size(n3)
        fftboxsize = round(n1_fft*n2_fft*n3_fft)
        maxG1,maxG2,maxG3=np.amax(np.abs(G[:,0])),np.amax(np.abs(G[:,1])),np.amax(np.abs(G[:,2]))
        #n4, n5, n6 = int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
        n4, n5, n6 = 10, 10, 80
        n4_fft, n5_fft, n6_fft = FFT_size(n4), FFT_size(n5), FFT_size(n6)
        fftboxsizeG = n4_fft*n5_fft*n6_fft
        # Première FFT
        print("Starting first FFT")
        chi0rG = np.zeros((nq, fftboxsizeG, ng), dtype = complex)
        phase_fac = np.ones((n4_fft, n5_fft, n6_fft), dtype = complex)
        ic = complex(0, 1)
        #print(ind_G_to_vec)
        for i in range(nq):
            q = ind_q_to_vec[i]
            q_vec = np.array([q[0]/nk[0], q[1]/nk[1], q[2]/nk[2]])
            #print(qvec)
            for m in range(n4_fft):
                for n in range(n5_fft):
                    for l in range(n6_fft):
                        phase_fac[m, n, l] = cmath.exp(ic*np.dot(q_vec, [m/n4_fft, n/n5_fft, l/n6_fft])) 
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

        # Seconde FFT
        print("Starting second FFT")
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
        chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size

        
        return chi0rr_out
          
    else:
        return "The second option is not valid, see the function definition to know the different possibilities"

def FFT_chi0_from_Mat(chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk, opt2 = "Kaltak"):
    if opt2 == "Standard":
        nq, ng1, ng2 = chi0GG.shape
        if ng1 != ng2:
            print("There is a problem in the code")
            return "There is a problem in the code"
        ng=ng1
        nqg = ng*nq
        n1, n2, n3 = round(n1), round(n2), round(n3)
        n1_fft, n2_fft, n3_fft = FFT_size(n1), FFT_size(n2), FFT_size(n3)
        fftboxsize = round(n1_fft*n2_fft*n3_fft)
        chi0rG = np.zeros((fftboxsize, nqg), dtype = complex)
        print("Starting first FFT")
        for i in range(nq):
            q_vec = ind_q_to_vec[i]
            q = [q_vec[0], q_vec[1], q_vec[2]]
            for j in range(ng): 
                FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)               
                for k in range(ng):
                    G_vec = ind_G_to_vec[k]
                    qG_vec_fft1 = [q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]]
                    FFTBox[round(qG_vec_fft1[0]), round(qG_vec_fft1[1]), round(qG_vec_fft1[2])] = chi0GG[i, j, k]
                FFT = np.fft.ifftn(FFTBox)
                chi0rG[:, j+i*ng] = np.reshape(FFT, fftboxsize)


        print("Starting second FFT")
        chi0rr = np.zeros((fftboxsize, n1_fft, n2_fft, n3_fft), dtype = complex)
        for i in range(fftboxsize):
            FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
            for j in range(nqg):
                qG_vec_fft2 = ind_qG_to_vec[j]
                FFTBox[-round(qG_vec_fft2[0]), -round(qG_vec_fft2[1]), -round(qG_vec_fft2[2])] = chi0rG[i, j]
            FFT = np.fft.ifftn(FFTBox) 
            chi0rr[i, :, :, :]=FFT
        chi0rr_out0 = np.reshape(chi0rr, (n1_fft, n2_fft, n3_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        return chi0rr_out

    elif opt2 == "Kaltak":
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
        print(n4_fft, n5_fft, n6_fft)
        fftboxsizeG = n4_fft*n5_fft*n6_fft
        # Première FFT
        print("Starting first FFT")
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
        elapsed1 = time.time()-start_time
        print(n4_fft, n5_fft, n6_fft )
        print("The first FFT took ", elapsed1, " seconds")
        # Seconde FFT
        print("Starting second FFT")
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
        print(n4_fft, n5_fft, n6_fft )
        chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        elapsed2 = time.time()-start_time-elapsed1
        print("The second FFT took ", elapsed2, " seconds")
        return chi0rr_out
          
    else:
        print("The second option is not valid, see the function definition to know the different possibilities")
        return [0]

def FFT_chi0_sizeadapt(chi0GG, ind_qG_to_vec, n1, n2, n3, n4, n5, n6, ind_q_to_vec, ind_G_to_vec, G, nk, vol, opt2 = "Kaltak"):
    if opt2 == "Standard":
        nq, ng1, ng2 = chi0GG.shape
        if ng1 != ng2:
            print("There is a problem in the code")
            return "There is a problem in the code"
        ng=ng1
        nqg = ng*nq
        n1, n2, n3 = round(n1), round(n2), round(n3)
        n1_fft, n2_fft, n3_fft = n1, n2, n3
        fftboxsize = round(n1_fft*n2_fft*n3_fft)
        chi0rG = np.zeros((fftboxsize, nqg), dtype = complex)
        print("Starting first FFT")
        for i in range(nq):
            q_vec = ind_q_to_vec[i]
            q = [q_vec[0], q_vec[1], q_vec[2]]
            for j in range(ng): 
                FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)               
                for k in range(ng):
                    G_vec = ind_G_to_vec[k]
                    qG_vec_fft1 = [q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]]
                    FFTBox[round(qG_vec_fft1[0]), round(qG_vec_fft1[1]), round(qG_vec_fft1[2])] = chi0GG[i, j, k]
                FFT = np.fft.ifftn(FFTBox)
                chi0rG[:, j+i*ng] = np.reshape(FFT, fftboxsize)


        print("Starting second FFT")
        chi0rr = np.zeros((fftboxsize, n1_fft, n2_fft, n3_fft), dtype = complex)
        for i in range(fftboxsize):
            FFTBox = np.zeros((n1_fft, n2_fft, n3_fft), dtype = complex)
            for j in range(nqg):
                qG_vec_fft2 = ind_qG_to_vec[j]
                FFTBox[-round(qG_vec_fft2[0]), -round(qG_vec_fft2[1]), -round(qG_vec_fft2[2])] = chi0rG[i, j]
            FFT = np.fft.ifftn(FFTBox) 
            chi0rr[i, :, :, :]=FFT
        chi0rr_out0 = np.reshape(chi0rr, (n1_fft, n2_fft, n3_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        return chi0rr_out

    elif opt2 == "Kaltak":
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
        print("Starting first FFT")
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
        elapsed1 = time.time()-start_time
        print(n4_fft, n5_fft, n6_fft )
        print("The first FFT took ", elapsed1, " seconds")
        # Seconde FFT
        print("Starting second FFT")
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
        elapsed2 = time.time()-start_time-elapsed1
        print("The second FFT took ", elapsed2, " seconds")
        return chi0rr_out
          
    else:
        print("The second option is not valid, see the function definition to know the different possibilities")
        return [0]
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
    
def Vis_tool_Bulk(chi0rr, R, A, B, C, nk, N=10000, isomin=1):
    
    # Point cloud

    n1,n2,n3=chi0rr[0, 0, 0, :, :, :].shape
    X = np.linspace(0, nk[0], n1, endpoint=False)
    Y = np.linspace(0, nk[1], n2, endpoint=False)
    Z = np.linspace(0, nk[2], n3, endpoint=False)
    Rx,Ry,Rz = R
    values = np.abs(np.real(chi0rr[Rx, Ry, Rz,]))
    points = pc.point_cloud(X, Y, Z, values, N, isomin)

    fn = RegularGridInterpolator((X, Y, Z), np.real(chi0rr[Rx, Ry, Rz,]), bounds_error=False, fill_value=None)
    color = fn(points)

    # Transformation of the point cloud
    xstart = points[:,0]
    ystart = points[:,1]
    zstart = points[:,2]  

    xend = A[0]*xstart+B[0]*ystart+zstart*C[0]
    yend = A[1]*xstart+B[1]*ystart+zstart*C[1]
    zend = A[2]*xstart+B[2]*ystart+zstart*C[2]

    # Lattice

    nk0 = nk[0]+1
    nk1 = nk[1]+1
    nk2 = nk[2]+1
    nat = nk0*nk1*nk2
    At_coord=np.zeros((nat, 3))
    count=0

    for i in range(nk0):
        for j in range(nk1):
            for k in range(nk2):
                At_coord[count] = [i, j, k]
                count += 1

    At_coordx = At_coord[:, 0]
    At_coordy = At_coord[:, 1]
    At_coordz = At_coord[:, 2]

    At_coordxf = At_coordx*A[0]+At_coordy*B[0]+At_coordz*C[0]
    At_coordyf = At_coordx*A[1]+At_coordy*B[1]+At_coordz*C[1]
    At_coordzf = At_coordx*A[2]+At_coordy*B[2]+At_coordz*C[2]
    At_coord_values=np.ones((nat))

    # Potential change 

    Rtx = [(Rx*A[0]+Ry*B[0]+Rz*C[0])*nk[0]/n1]
    Rty = [(Rx*A[1]+Ry*B[1]+Rz*C[1])*nk[1]/n2]
    Rtz = [(Rx*A[2]+Ry*B[2]+Rz*C[2])*nk[2]/n3]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=xend, y=yend, z=zend,
        mode='markers',
        marker=dict(
            size=2,
            color=color, 
            colorscale='Viridis',
            colorbar=dict(thickness=20)
        )
    )])
    fig.add_trace(go.Scatter3d(
        x=At_coordxf, y=At_coordyf, z=At_coordzf,
        mode='markers',
        marker=dict(
            size=5,
            color=At_coord_values, 
            colorscale=['rgb(0,0,0)', 'rgb(0,0,0)']
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=Rtx, y=Rty, z=Rtz,
        mode='markers',
        marker=dict(
            size=7,
            color=[1], 
            colorscale=['rgb(255,0,0)', 'rgb(255,0,0)']
        )
    ))

    fig.show()

def Vis_tool(chi0rr, R, A, B, C, nk, nat_cell, pos_red, N=10000, isomin=1):
    
    # Point cloud

    n1,n2,n3=chi0rr[0, 0, 0, :, :, :].shape
    X = np.linspace(0, nk[0], n1, endpoint=False)
    Y = np.linspace(0, nk[1], n2, endpoint=False)
    Z = np.linspace(0, nk[2], n3, endpoint=False)
    Rx,Ry,Rz = int(R[0]), int(R[1]), int(R[2])
    values = np.abs(np.real(chi0rr[Rx, Ry, Rz,]))
    points = pc.point_cloud(X, Y, Z, values, N, isomin)
    
    fn = RegularGridInterpolator((X, Y, Z), np.real(chi0rr[Rx, Ry, Rz,]), bounds_error=False, fill_value=None)
    color = fn(points)

    # Transformation of the point cloud
    xstart = points[:,0]
    ystart = points[:,1]
    zstart = points[:,2]  

    xend = A[0]*xstart+B[0]*ystart+zstart*C[0]
    yend = A[1]*xstart+B[1]*ystart+zstart*C[1]
    zend = A[2]*xstart+B[2]*ystart+zstart*C[2]

    # Lattice

    nk0 = nk[0]+1
    nk1 = nk[1]+1
    nk2 = nk[2]+1
    nat = nat_cell*nk0*nk1*nk2
    pos_cart=np.zeros((nat_cell,3))
    At_coord=np.zeros((nat, 3))
    count=0

    R_vec=np.array([A,B,C])
    for i in range(nat_cell):
        for j in range(3):
            pos_cart[i, j] = np.matmul(pos_red[i,:], R_vec[:,j])

    for n in range(nat_cell):
        for i in range(nk2):
            for j in range(nk0):
                for k in range(nk1):
                    At_coord[count] = pos_cart[n]+j*R_vec[0]+k*R_vec[1]+i*R_vec[2]
                    count += 1

    At_coordxf = At_coord[:, 0]
    At_coordyf = At_coord[:, 1]
    At_coordzf = At_coord[:, 2]

    At_coord_values=np.ones(nat)

    # Potential change 
    
    Rtx = [(Rx*A[0]+Ry*B[0]+Rz*C[0])*nk[0]/n1]
    Rty = [(Rx*A[1]+Ry*B[1]+Rz*C[1])*nk[1]/n2]
    Rtz = [(Rx*A[2]+Ry*B[2]+Rz*C[2])*nk[2]/n3]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=xend, y=yend, z=zend,
        mode='markers',
        marker=dict(
            size=2,
            color=color, 
            colorscale='Viridis',
            colorbar=dict(thickness=20)
        )
    )])
    fig.add_trace(go.Scatter3d(
        x=At_coordxf, y=At_coordyf, z=At_coordzf,
        mode='markers',
        marker=dict(
            size=5,
            color=At_coord_values, 
            colorscale=['rgb(0,0,0)', 'rgb(0,0,0)']
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=Rtx, y=Rty, z=Rtz,
        mode='markers',
        marker=dict(
            size=7,
            color=[1], 
            colorscale=['rgb(255,0,0)', 'rgb(255,0,0)']
        )
    ))

    fig.show()

def Supercell_Vis(A, B, C, nk, nat_cell, pos_red):
    nk0 = nk[0]
    nk1 = nk[1]
    nk2 = nk[2]
    nat = nat_cell*nk0*nk1*nk2
    pos_cart=np.zeros((nat_cell,3))
    At_coord=np.zeros((nat, 3))
    count=0

    R_vec=np.array([A,B,C])
    for i in range(nat_cell):
        for j in range(3):
            pos_cart[i, j] = np.matmul(pos_red[i,:], R_vec[:,j])

    for n in range(nat_cell):
        for i in range(nk2):
            for j in range(nk0):
                for k in range(nk1):
                    At_coord[count] = pos_cart[n]+j*R_vec[0]+k*R_vec[1]+i*R_vec[2]
                    count += 1

    At_coordxf = At_coord[:, 0]
    At_coordyf = At_coord[:, 1]
    At_coordzf = At_coord[:, 2]

    At_coord_values=np.ones(nat)
    fig = go.Figure(data=[go.Scatter3d(
        x=At_coordxf, y=At_coordyf, z=At_coordzf,
        mode='markers',
        marker=dict(
            size=6,
            color=At_coord_values, 
            colorscale='matter'
        )
    )])
    fig.show()

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

def Friedel_Oscillation(x, rho_0, center, freq):
    return rho_0+math.cos(2*math.pi*freq*x+center)/x**3

def MatCharac(Matrr, MatGG, filename, input_filename, opt1 = 'FullBZ', opt2 = 'Standart'):
    MatCharac = open(filename, mode='w')
    MatCharac.write("The matrix chi^0(r,r') was obtained from the file" + input_filename + " with the options "+ opt1 +" and "+ opt2)
    MatCharac.write("\n\n###################################")
    MatCharac.write("\n###################################")
    MatCharac.write("\n\nThe matrix chi^0(q+G, q+G') has the following characterisitcs :")
    MatCharac.write("\n\n- The matrix has a shape " + str(MatGG.shape))
    MatCharac.write("\n\n- The matrix has a size " + str(MatGG.size))
    MatCharac.write("\n\n- chi^0(0, 0) = " +str(MatGG[0, 0, 0]))
    MatCharac.write("\n\n- The max abs real value of chi^0 is " + str(np.amax(np.abs(np.real(MatGG)))))
    MatCharac.write("\n\n- The max abs imag value of chi^0 is " + str(np.amax(np.abs(np.imag(MatGG)))))
    sum1 = np.sum(MatGG)
    MatCharac.write("\n\nSum over all components of chi^0  = " + str(sum1))
    MatCharac.write("\n\n###################################")
    MatCharac.write("\n###################################")
    MatCharac.write("\n\nThe matrix chi^0(r, r') has the following characterisitcs :")
    MatCharac.write("\n\n- The matrix has a shape " + str(Matrr.shape))
    MatCharac.write("\n\n- The matrix has a size " + str(Matrr.size))
    MatCharac.write("\n\n- chi^0(0, 0) = " +str(Matrr[0, 0, 0, 0, 0, 0]))
    MatCharac.write("\n\n- The max abs real value of chi^0 is " + str(np.amax(np.abs(np.real(Matrr)))))
    MatCharac.write("\n\n- The max abs imag value of chi^0 is " + str(np.amax(np.abs(np.imag(Matrr)))))
    sum1 = np.sum(Matrr)
    MatCharac.write("\n\nSum over all components of chi^0  = " + str(sum1))
    MatCharac.close()

""" for i in range(nq):
            q = ind_q_to_vec[i]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                q_origin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
            else:
                SymR = SymRec[SymData[0]]
                q_origin = kpoints[SymData[1]]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            print("qpoint #",i,"/",nq)
            for j in range(ng):
                G1 = ind_G_to_vec[j]
                qG1_vec = np.round([q[0]+G1[0], q[1]+G1[1], q[2]+G1[2]])
                qG1 = (qG1_vec[0], qG1_vec[1], qG1_vec[2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                SG1 = np.round(np.matmul(np.linalg.inv(SymR), G1))
                indchi1 = vec_G_to_ind[(SG1[0], SG1[1], SG1[2])]
                for k in range(ng):
                    G2 = ind_G_to_vec[k]
                    qG2_vec = np.round([q[0]+G2[0], q[1]+G2[1], q[2]+G2[2]])
                    qG2 = (qG2_vec[0], qG2_vec[1], qG2_vec[2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        SG2 = np.round(np.matmul(np.linalg.inv(SymR), G2))
                        indchi2 = vec_G_to_ind[(SG2[0], SG2[1], SG2[2])]
                        chi0GG[i, j, k] = chi0[omega, indchi1, indchi2] """

##########chi0(q, g1, g2) = chi0*(q, g2, g1) --> f is hermitian
""" for i in range(nkpt):
    for j in range(ng):
        for k in range(ng):
            chi0GG[i,j,k] = 1/2*(chi0GG[i,j,k] + np.conj(chi0GG[i,k,j]))
            chi0GG[i,k,j] = 1/2*(np.conj(chi0GG[i,j,k]) + chi0GG[i,k,j]) """

##############chi0(q, g1, g2) = chi0*(-q, -g1, -g2) --> f is real
""" for i in range(nq):
    q = ind_q_to_vec[i]
    qopp = [-q[0], -q[1], -q[2]]
    if qopp[0] == round(-0.5*nk[0]):
        qopp[0] = -qopp[0]
    if qopp[1] == round(-0.5*nk[1]):
    qopp[1] = -qopp[1]
    if qopp[2] == round(-0.5*nk[2]):
        qopp[2] = -qopp[2]
    l = vec_q_to_ind[(qopp[0], qopp[1], qopp[2])]
    for j in range(ng):
        G_vec1 = ind_G_to_vec[j]
        m = vec_G_to_ind[(-G_vec1[0], -G_vec1[1], -G_vec1[2])]
        for k in range(ng):
            G_vec2 = ind_G_to_vec[k]
            n = vec_G_to_ind[(-G_vec2[0], -G_vec2[1], -G_vec2[2])]
            chi0GG[i,j,k] = 1/2*(chi0GG[i,j,k] + np.conj(chi0GG[l,m,n]))
            chi0GG[l,m,n] = 1/2*(np.conj(chi0GG[i,j,k]) + chi0GG[l,m,n]) """

##############chi0(q, g1, g2) = chi0(-q, -g2, -g1) --> f is symmetric
""" for i in range(nq):
    q = ind_q_to_vec[i]
    qopp = [-q[0], -q[1], -q[2]]
    if qopp[0] == round(-0.5*nk[0]):
        qopp[0] = -qopp[0]
    if qopp[1] == round(-0.5*nk[1]):
        qopp[1] = -qopp[1]
    if qopp[2] == round(-0.5*nk[2]):
        qopp[2] = -qopp[2]
    l = vec_q_to_ind[(qopp[0], qopp[1], qopp[2])]
    for j in range(ng):
        G_vec1 = ind_G_to_vec[j]
        m = vec_G_to_ind[(-G_vec1[0], -G_vec1[1], -G_vec1[2])]
        for k in range(ng):
            G_vec2 = ind_G_to_vec[k]
            n = vec_G_to_ind[(-G_vec2[0], -G_vec2[1], -G_vec2[2])]
            chi0GG[i,j,k] = chi0GG[l, n, m] = 1/2*(chi0GG[i,j,k] + chi0GG[l, n, m]) """

##########chi0(q, g1, g2) = chi0(-q, -g2, -g1)
""" for i in range(nq):
    q = ind_q_to_vec[i]
    qopp = [-q[0], -q[1], -q[2]]
    if qopp[0] == round(-0.5*nk[0]):
        qopp[0] = -qopp[0]
    if qopp[1] == round(-0.5*nk[1]):
        qopp[1] = -qopp[1]
    if qopp[2] == round(-0.5*nk[2]):
        qopp[2] = -qopp[2]
    l = vec_q_to_ind[(qopp[0], qopp[1], qopp[2])]
    for j in range(ng):
        G_vec1 = ind_G_to_vec[j]
        m = vec_G_to_ind[(-G_vec1[0], -G_vec1[1], -G_vec1[2])]
        for k in range(ng):
            G_vec2 = ind_G_to_vec[k]
            n = vec_G_to_ind[(-G_vec2[0], -G_vec2[1], -G_vec2[2])]
            chi0GG[i,j,k] = chi0GG[l,m,n] = 1/2*(chi0GG[i,j,k] + np.conj(chi0GG[l,m,n])) """

##########chi0(q, g1, g2) = chi0(q, g2, g1)
""" for i in range(nq):
    for j in range(ng):
        for k in range(ng):
            chi0GG[i,j,k] = chi0GG[i,k,j] = 1/2*(chi0GG[i,j,k] + chi0GG[i,k,j]) """

##########chi0(q, g1, g2) = chi0^*(q, g2, g1)   
""" for i in range(nq):
    for j in range(ng):
        for k in range(ng):
            chi0GG[i,j,k] = chi0GG[i,k,j] = 1/2*(chi0GG[i,j,k] + np.conj(chi0GG[i,k,j])) """

##########chi0(q, g1, g2) is real
#chi0GG = np.real(chi0GG)


"""     complete_dict_sym = {}
        inv_dict = {}
        count = 0
        for i in range(nqibzG):
            qibzG_vec = ind_qibzG_to_vec[i]
            for j in range(nsym):
                qG_rot = np.round(np.matmul(SymRec[j],[qibzG_vec[0], qibzG_vec[1], qibzG_vec[2]]))
                if (qG_rot[0], qG_rot[1], qG_rot[2]) in complete_dict_sym.keys():
                    complete_dict_sym[(qG_rot[0], qG_rot[1], qG_rot[2])] = complete_dict_sym[(qG_rot[0], qG_rot[1], qG_rot[2])]+1
                else:
                    complete_dict_sym[(qG_rot[0], qG_rot[1], qG_rot[2])] = 0
                    inv_dict[count] = (qG_rot[0], qG_rot[1], qG_rot[2])
                    count+=1
        count_q = 0
        for i in range(nqibzG):
            qibzG_vec = ind_qibzG_to_vec[i]
            if qibzG_vec not in vec_with_missing_sym.keys():
                vec_with_all_sym[qibzG_vec] = count_q
                count_q += 1
        print(count_q)
        print(len(vec_with_missing_sym))
        qG_rot_miss = {}
        for qG_vec in vec_with_missing_sym.keys():
            for j in range(nsym):
                qG_rot = np.round(np.matmul(SymRec[j],[qG_vec[0], qG_vec[1], qG_vec[2]]))
                qG_rot_miss[(qG_rot[0], qG_rot[1], qG_rot[2])] = 0
        
        print(qG_rot_miss)
        qG_rot_in = {}
        for qG_vec in vec_with_all_sym.keys():
            for j in range(nsym):
                qG_rot = np.round(np.matmul(SymRec[j],[qG_vec[0], qG_vec[1], qG_vec[2]]))
                qG_rot_in[(qG_rot[0], qG_rot[1], qG_rot[2])] = 0
        print(qG_rot_in)
        print(len(qG_rot_in), len(qG_rot_miss))
        count_overlap = 0
        for qG_vec in qG_rot_miss.keys():
            if qG_vec in qG_rot_in.keys():
                count_overlap += 1
        for qG_vec in qG_rot_in.keys():
            if qG_vec in qG_rot_miss.keys():
                count_overlap += 1
        print(count_overlap)
        
        
        
        vec_qG_to_ind_without_border, ind_qG_to_vec_without_border = {}, {}
        count2 = 0
        for i in range(count):
            qG_vec = ind_qGrot_to_vec[i]
            if (-qG_vec[0], -qG_vec[1], -qG_vec[2]) not in vec_qGrot_to_ind.keys():
                continue
            else:
               vec_qG_to_ind_without_border[qG_vec] = count2
               ind_qG_to_vec_without_border[count2] = qG_vec
               count2 += 1 
        
        ind_qG_to_ind_SqG = {}       
        for i in range(nq):
            q = ind_q_to_vec[i]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                q_origin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
            else:
                SymR = SymRec[SymData[0]]
                q_origin = kpoints[SymData[1]]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                qG_vec = ind_qbzG_to_vec[i*ng+j]
                if qG_vec not in vec_qG_to_ind_without_border.keys():
                    continue
                else:
                    SG = np.round(np.matmul(np.linalg.inv(SymR), G_vec))
                    if (q_origin[0]+SG[0], q_origin[1]+SG[1], q_origin[2]+SG[2]) in vec_with_missing_sym.keys():
                        print("there is a problem")
                    indchi = vec_G_to_ind[(SG[0], SG[1], SG[2])]
                    ind_qG_to_ind_SqG[i*ng+j] = indchi
        count_off = 0            
        for i in range(nq):
            q = ind_q_to_vec[i]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                q_ind = SymRec[2][1]
                #t=TRec[SymData[2][0]]
            else:
                SymR = SymRec[SymData[0]]
                q_ind = SymData[1]
            #chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            #print("qpoint #", str(i+1), "/", nq)
            for j in range(ng):
                if j+i*ng not in ind_qG_to_ind_SqG.keys():
                    continue
                else:
                    indchi1 = ind_qG_to_ind_SqG[j+i*ng]
                    for k in range(ng):
                        if k+i*ng not in ind_qG_to_ind_SqG.keys():
                            continue
                        else:
                            indchi2 = ind_qG_to_ind_SqG[k+i*ng]
                            if smallchi0GG[q_ind, indchi1, indchi2] == 0:
                                count_off += 1
                            chi0GG[i, j, k] = smallchi0GG[q_ind, indchi1, indchi2]#*cmath.exp(ic*np.dot(t, G2-G1))
        print(count_off)  
        for i in range(nq):
            q = ind_q_to_vec[i]
            qopp = [-q[0], -q[1], -q[2]]
            if qopp[0] == round(-0.5*nk[0]):
                qopp[0] = -qopp[0]
            if qopp[1] == round(-0.5*nk[1]):
                qopp[1] = -qopp[1]
            if qopp[2] == round(-0.5*nk[2]):
                qopp[2] = -qopp[2]
            l = vec_q_to_ind[(qopp[0], qopp[1], qopp[2])]
            for j in range(ng):
                G_vec1 = ind_G_to_vec[j]
                m = vec_G_to_ind[(-G_vec1[0], -G_vec1[1], -G_vec1[2])]
                for k in range(ng):
                    G_vec2 = ind_G_to_vec[k]
                    n = vec_G_to_ind[(-G_vec2[0], -G_vec2[1], -G_vec2[2])]
                    chi0GG[i,j,k] = chi0GG[l, n, m] = 1/2*(chi0GG[i,j,k] + chi0GG[l, n, m])"""
"""         elif opt=='FromSym':
        
        structure = abipy.core.structure.Structure.from_file(filename)
        Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
        SymRec = Sym.symrec
        nsym = len(SymRec)
        nk = fsk(kpoints)
        #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ. Needs to pay attention to rounding errors+needs to use Umklapp vectors to get all the data

        vec_q_to_ind, ind_q_to_vec, sym_dict = {},{},{}
        ind = 0
        for i in range(nsym):
            for j in range(nkpt):
                q = np.round(np.matmul(SymRec[i], kpoints[j].frac_coords), 3)#+TRec[i]
                if np.amax(q) > 0.5 or np.amin(q) <= - 0.5:
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
                #print(q_in_bz)
                q_vec = np.round(np.multiply([q_in_bz[0], q_in_bz[1], q_in_bz[2]], nk))
                if (q_vec[0], q_vec[1], q_vec[2]) not in vec_q_to_ind.keys():
                            ind_q_to_vec[ind] = (q_vec[0], q_vec[1], q_vec[2])
                            vec_q_to_ind[(q_vec[0], q_vec[1], q_vec[2])] = ind
                            sym_dict[ind] = (i, j, (0, 0))
                            ind+=1
                else:
                    continue

        #Verification de l'inclusion de la symétrie d'inversion
        invsym_bool = IsInvSymIn(SymRec, nsym)
        
        if invsym_bool==False:
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
        #print(len(ind_q_to_vec))
        nq = len(sym_dict)
        #Liste des vecteurs G
        vec_G_to_ind = {}
        ind_G_to_vec = {}
        for i in range(ng):
            G_vec = np.round(np.multiply([G[i, 0], G[i, 1], G[i, 2]], nk))
            vec_G_to_ind[(G_vec[0], G_vec[1], G_vec[2])] = i
            ind_G_to_vec[i] = (G_vec[0], G_vec[1], G_vec[2])
        
        #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau)
        vec_qibzG_to_ind = {}
        ind_qibzG_to_vec = {}
        vec_qbzG_to_ind = {}
        ind_qbzG_to_vec ={}

        nvec = nq*ng
        vec_table = np.zeros((nvec,3), dtype = int)
        for i in range(nkpt):
            for j in range(ng):
                kpt = kpoints[i].frac_coords
                G_vec = G[j]
                qG = np.round(np.multiply([kpt[0]+G_vec[0], kpt[1]+G_vec[1], kpt[2]+G_vec[2]], nk))
                vec_qibzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qibzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
                ind = j + i * ng
                vec_table[ind] = np.round(np.multiply((kpoints[i].frac_coords + G[j]), nk))
        nqibzG = nkpt*ng

        for i in range(nq):
            q = ind_q_to_vec[i]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                qG = np.round([q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]])
                vec_qbzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qbzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
        print(len(vec_qbzG_to_ind))
        #Test des vecteurs dans le sets après rotation
        vec_qGrot_to_ind = {}
        ind_qGrot_to_vec = {}
        
        count = 0
        print("Basic Dict initialized")
        for i in range(nqibzG):
            qG_vec = ind_qibzG_to_vec[i]
            for j in range(nsym):
                qG_rot = np.round(np.matmul(SymRec[j],[qG_vec[0], qG_vec[1], qG_vec[2]]))
                #print(qG_rot)
                if (qG_rot[0], qG_rot[1], qG_rot[2]) not in vec_qbzG_to_ind.keys() or (qG_rot[0], qG_rot[1], qG_rot[2]) in vec_qGrot_to_ind.keys():
                    continue
                else:
                    vec_qGrot_to_ind[(qG_rot[0], qG_rot[1], qG_rot[2])] = count
                    ind_qGrot_to_vec[count] = (qG_rot[0], qG_rot[1], qG_rot[2])
                    count += 1
        #Liste finale des vecteurs q+G valables
        print("Dict with symmetry initialized")
        vec_qG_to_ind_without_border, ind_qG_to_vec_without_border = {}, {}
        count2 = 0
        for i in range(count):
            qG_vec = ind_qGrot_to_vec[i]
            if (-qG_vec[0], -qG_vec[1], -qG_vec[2]) not in vec_qGrot_to_ind.keys():
                continue
            else:
               vec_qG_to_ind_without_border[qG_vec] = count2
               ind_qG_to_vec_without_border[count2] = qG_vec
               count2 += 1
        print("qGvec sorted")
        nvec = len(vec_qG_to_ind_without_border) 
        vec_table_without_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]
        

        ind_qG_to_ind_SqG = {}
        for i in range(nq):
            q = ind_q_to_vec[i]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                q_origin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
            else:
                SymR = SymRec[SymData[0]]
                q_origin = kpoints[SymData[1]]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                qG_vec = ind_qbzG_to_vec[i*ng+j]
                if qG_vec not in vec_qG_to_ind_without_border.keys():
                    continue
                else:
                    SG = np.round(np.matmul(np.linalg.inv(SymR), G_vec))
                    indchi = vec_G_to_ind[(SG[0], SG[1], SG[2])]
                    ind_qG_to_ind_SqG[i*ng+j] = indchi
        print("Dict of symmetry initialized")



        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:, 1])), np.amax(np.abs(vec_table_without_border[:, 2]))
        n1, n2, n3=(2*s1)+1, (2*s2)+1, (2*s3)+1
        chi0GG = np.zeros((nq, ng, ng), dtype = complex)

        for i in range(nq):
            q = ind_q_to_vec[i]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                q_origin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
            else:
                SymR = SymRec[SymData[0]]
                q_origin = kpoints[SymData[1]]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            #print("qpoint #", str(i+1), "/", nq)
            for j in range(ng):
                if j+i*ng not in ind_qG_to_ind_SqG.keys():
                    continue
                else:
                    indchi1 = ind_qG_to_ind_SqG[j+i*ng]
                    for k in range(ng):
                        if k+i*ng not in ind_qG_to_ind_SqG.keys():
                            continue
                        else:
                            indchi2 = ind_qG_to_ind_SqG[k+i*ng]
                            chi0GG[i, j, k] = chi0[omega, indchi1, indchi2]#*cmath.exp(ic*np.dot(t, G2-G1))


         """
""" def Sym_chi0GG0(filename, omega = 0):
    sus_ncfile, kpoints, ng, nkpt, G = openfile(filename)
    structure = abipy.core.structure.Structure.from_file(filename)
    Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
    #dict_struct = structure.to_abivars()
    SymRec = Sym.symrec
    print(len(SymRec))
        #Trec = Sym.tnons
    nsym = len(SymRec)
        #nsym = round(len(SymRec)/natom)
    nk = fsk(kpoints)
    vec_q_to_ind, ind_q_to_vec, sym_dict = {},{},{}
    ind = 0
    qibzvec_to_ind  = {}
    for i in range(nkpt):
        q = np.round(np.multiply(kpoints[i].frac_coords, nk))
        qibzvec_to_ind[(q[0], q[1], q[2])] = i
        vec_qibz_to_ind[i] = (q[0], q[1], q[2])
    for i in range(nsym):
        for j in range(nkpt):
            q = np.round(np.matmul(SymRec[i], kpoints[j].frac_coords), 3)#+TRec[i]
            if np.amax(q) > 0.5 or np.amin(q) <= - 0.5:
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
                #print(q_in_bz)
            q_vec = np.round(np.multiply([q_in_bz[0], q_in_bz[1], q_in_bz[2]], nk))
            if (q_vec[0], q_vec[1], q_vec[2]) not in vec_q_to_ind.keys():
                        ind_q_to_vec[ind] = (q_vec[0], q_vec[1], q_vec[2])
                        vec_q_to_ind[(q_vec[0], q_vec[1], q_vec[2])] = ind
                        sym_dict[ind] = (i, j, (0, 0))
                        ind+=1
            else:
                continue
    nq = len(sym_dict)
        #Liste des vecteurs G
    vec_G_to_ind = {}
    ind_G_to_vec = {}
    for i in range(ng):
        G_vec = np.round(np.multiply([G[i, 0], G[i, 1], G[i, 2]], nk))
        vec_G_to_ind[(G_vec[0], G_vec[1], G_vec[2])] = i
        ind_G_to_vec[i] = (G_vec[0], G_vec[1], G_vec[2])
        #print(ng)
        #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau)
    vec_qibzG_to_ind = {}
    ind_qibzG_to_vec = {}
    vec_qbzG_to_ind = {}
    ind_qbzG_to_vec ={}

    for i in range(nkpt):
        for j in range(ng):
            kpt = kpoints[i].frac_coords
            G_vec = G[j]
            qG = np.round(np.multiply([kpt[0]+G_vec[0], kpt[1]+G_vec[1], kpt[2]+G_vec[2]], nk))
            vec_qibzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
            ind_qibzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
        
    nqibzG = nkpt*ng
    vec_table = np.zeros((nq*ng, 3))
    for i in range(nq):
        q = ind_q_to_vec[i]
        for j in range(ng):
            G_vec = ind_G_to_vec[j]
            qG = np.round([q[0]+G_vec[0], q[1]+G_vec[1], q[2]+G_vec[2]])
            vec_qbzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
            ind_qbzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
            vec_table[j+i*ng] = qG

    smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)
    for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega]

    chi0GGsym = np.zeros((nkpt, ng, ng), dtype = complex)
    comb_vec = {}
    for i in range(nkpt):
        q = ind_q_to_vec[i]
        qvec = [q[0], q[1], q[2]]
        for j in range(ng):
            G1 = ind_G_to_vec[j]
            G1vec = [G1[0], G1[1], G1[2]]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                G2vec = [G2[0], G2[1], G2[2]]
                if (i, j, k) not in comb_vec.keys():
                    sum_chi = 0
                    sym_vec = {}
                    count = 0
                    for l in range(nsym):
                        S = SymRec[l]
                        Sq = np.matmul(S, qvec)
                        SG1 = np.matmul(S, G1vec)
                        SG2 = np.matmul(S, G2vec)
                        if (Sq[0], Sq[1], Sq[2]) in qibzvec_to_ind.keys() and (SG1[0], SG1[1], SG1[2]) in vec_G_to_ind.keys() and (SG2[0], SG2[1], SG2[2]) in vec_G_to_ind.keys():
                            indq = vec_q_to_ind[(Sq[0], Sq[1], Sq[2])]
                            indG1 = vec_G_to_ind[(SG1[0], SG1[1], SG1[2])]
                            indG2 = vec_G_to_ind[(SG2[0], SG2[1], SG2[2])]
                            sum_chi += smallchi0GG[indq, indG1, indG2]
                            sym_vec[(indq, indG1, indG2)] = l
                            comb_vec[(indq, indG1, indG2)] = 0
                            count += 1
                    mean = sum_chi/count
                    for (m, n, o) in sym_vec.keys():
                        chi0GGsym[m, n, o] = mean
    return chi0GGsym """
""" def Sym_chi0GG0(sus_ncfile, kpoints, ng, nkpt, G, SymRec, nsym, ind_q_to_vec, ind_G_to_vec, qibzvec_to_ind, vec_G_to_ind, vec_qibzG_to_ind, ind_qibzG_to_vec, omega):
    # Function that symmetrizes the chi^0GG output of Abinit such that two symmetric values have exactly the same values
    smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)
    for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega]

    chi0GGsym = np.zeros((nkpt, ng, ng), dtype = complex)
    #Dictionnary to store all the q, G, G' set already treated
    comb_vec = {}
    for i in range(nkpt):
        q = ind_q_to_vec[i]
        qvec = [q[0], q[1], q[2]]
        for j in range(ng):
            G1 = ind_G_to_vec[j]
            G1vec = [G1[0], G1[1], G1[2]]
            for k in range(ng):
                G2 = ind_G_to_vec[k]
                G2vec = [G2[0], G2[1], G2[2]]
                if (i, j, k) not in comb_vec.keys():
                    sum_chi = 0
                    #Dictionnary to store all the symmetrical q,G,G' sets
                    sym_vec = {}
                    count = 0
                    for l in range(nsym):
                        S = SymRec[l]
                        Sq = np.matmul(S, qvec)
                        SG1 = np.matmul(S, G1vec)
                        SG2 = np.matmul(S, G2vec)
                        if (Sq[0], Sq[1], Sq[2]) in qibzvec_to_ind.keys() and (SG1[0], SG1[1], SG1[2]) in vec_G_to_ind.keys() and (SG2[0], SG2[1], SG2[2]) in vec_G_to_ind.keys():
                            indq = qibzvec_to_ind[(Sq[0], Sq[1], Sq[2])]
                            indG1 = vec_G_to_ind[(SG1[0], SG1[1], SG1[2])]
                            indG2 = vec_G_to_ind[(SG2[0], SG2[1], SG2[2])]
                            sum_chi += smallchi0GG[indq, indG1, indG2]
                            sym_vec[(indq, indG1, indG2)] = l
                            comb_vec[(indq, indG1, indG2)] = 0
                            count += 1
                    mean = sum_chi/count
                    #Replacement off all the values in the set by the mean of all the values
                    for (m, n, o) in sym_vec.keys():
                        chi0GGsym[m, n, o] = mean
    return chi0GGsym
 """
""" if l1%2==0:
        if l2%2==0:
            charge_cut[0:l3, 0:l4] = charge_cut_raw[l3:l1, l4:l2]
            charge_cut[l3:l1, 0:l4] = charge_cut_raw[0:l3, l4:l2]
            charge_cut[0:l3, l4:l2] = charge_cut_raw[l3:l1, 0:l4]
            charge_cut[l3:l1, l4:l2] = charge_cut_raw[0:l3, 0:l4]
        else:
            charge_cut[0:l3, 0:l4] = charge_cut_raw[l3:l1, l4+1:l2]
            charge_cut[l3:l1, 0:l4] = charge_cut_raw[0:l3, l4+1:l2]
            charge_cut[0:l3, l4:l2] = charge_cut_raw[l3:l1, 0:l4+1]
            charge_cut[l3:l1, l4:l2] = charge_cut_raw[0:l3, 0:l4+1]
    else:
        if l2%2==0:
            charge_cut[0:l3, 0:l4] = charge_cut_raw[l3+1:l1, l4:l2]
            charge_cut[l3:l1, 0:l4] = charge_cut_raw[0:l3+1, l4:l2]
            charge_cut[0:l3, l4:l2] = charge_cut_raw[l3+1:l1, 0:l4]
            charge_cut[l3:l1, l4:l2] = charge_cut_raw[0:l3+1, 0:l4]
        else:
            charge_cut[0:l3, 0:l4] = charge_cut_raw[l3+1:l1, l4+1:l2]
            charge_cut[l3:l1, 0:l4] = charge_cut_raw[0:l3+1, l4+1:l2]
            charge_cut[0:l3, l4:l2] = charge_cut_raw[l3+1:l1, 0:l4+1]
            charge_cut[l3:l1, l4:l2] = charge_cut_raw[0:l3+1, 0:l4+1] """
""" 
    if l1%2==0 and l2%2==0:
        charge_cut = charge_cut_inter
    elif l1%2==1 and l2%2==0:
        charge_cut = np.zeros((l1+1, l2), dtype = complex)
        charge_cut[1:l1+1] = charge_cut_inter
        charge_cut[0]=charge_cut_inter[l1-1]
        l1+=1
    elif l1%2==0 and l2%2==1:
        charge_cut = np.zeros((l1, l2+1), dtype = complex)
        charge_cut[:,0:l2] = charge_cut_inter
        charge_cut[:,l2]=charge_cut_inter[:, 0]
    else:
        charge_cut = np.zeros((l1+1, l2+1), dtype = complex)
        charge_cut[0:l1,0:l2] = charge_cut_inter
        charge_cut[l1, 0:l2]=charge_cut_inter[0]
        charge_cut[:,l2]=charge_cut[:, 0] """