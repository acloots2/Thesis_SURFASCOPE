######Xavier's Algorithm#########
from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
import abipy
import numpy as np
import cmath
import pointcloud as pc
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go

def Build_Chi0GG(filename, opt, omega = 0):
    sus_ncfile, kpoints, nw, ng, nkpt, G, nqg = openfile(filename)
    if opt == 'FullBZ':
        nvec = nkpt*ng
        nk = fsk(kpoints)
        #Creation d'un dictionnaire :
        vec_qG_to_ind = {}
        ind_qG_to_vec = {}
        ind_q_to_vec = {}
        ind_G_to_vec = {}
        vec_table = np.zeros((nvec,3), dtype = int)

        for i in range(nkpt):
            q = kpoints[i].frac_coords
            ind_q_to_vec[i] = np.multiply([q[0], q[1], q[2]], nk)
            for j in range(ng):
                if i == 0:
                    G_vec = G[j]
                    ind_G_to_vec[j] = np.multiply([G_vec[0], G_vec[1], G_vec[2]], nk)
                ind = j + i * ng
                vec_table[ind] = np.round(np.multiply((kpoints[i].frac_coords + G[j]), nk))
                qG = (vec_table[ind,0], vec_table[ind,1], vec_table[ind,2])
                vec_qG_to_ind[qG] = ind 
                ind_qG_to_vec[ind] = qG

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

        vec_table_without_border = np.zeros((count, 3), dtype = int)
        for i in range(count):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:,1])), np.amax(np.abs(vec_table_without_border[:,2]))
        n1, n2, n3 = 2*s1+1, 2*s2+1, 2*s3+1

        #Initialisation of chi0 :

        chi0GG = np.zeros((nkpt, ng, ng), dtype = complex)

        for i in range(nkpt):
            chi0 = sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat
            chi0GG[i, :, :] = chi0[omega]
            q = np.multiply(kpoints[i].frac_coords, nk)
            for j in range(ng):
                qG_vec = q+np.multiply(G[j], nk)
                qG=(qG_vec[0], qG_vec[1], qG_vec[2])
                if qG not in vec_qG_to_ind_without_border.keys():
                    print()
                    chi0GG[i, j, :] = np.zeros(ng)
                    chi0GG[i, :, j] = np.zeros(ng)
        #TO DO : 
        # Add Symmetrization
        return chi0GG, vec_qG_to_ind_without_border, ind_qG_to_vec_without_border, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G
    
    elif opt=='FromSym':
        
        structure = abipy.core.structure.Structure.from_file(filename)
        Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
        SymRec = Sym.symrec
        TRec = Sym.tnons
        Tim_Rev = Sym.has_timerev
        nsym = len(SymRec)
        nk = fsk(kpoints)
        #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ

        vec_q_toind, ind_q_tovec, sym_dict = {},{},{}
        ind_q_to_vec = {}
        ind = 0
        for i in range(nsym):
            for j in range(nkpt):
                q = np.round(np.matmul(SymRec[i], kpoints[j].frac_coords),5)#+TRec[i]
                if (q[0], q[1], q[2]) not in vec_q_toind.keys():
                    if np.amax(q) <= 0.5 and np.amin(q) >- 0.5:
                        vec_q_toind[(q[0], q[1], q[2])] = ind
                        ind_q_tovec[ind] = (q[0], q[1], q[2])
                        ind_q_to_vec[ind] = np.multiply(q, nk)
                        sym_dict[ind]=(i, j, (0, 0))
                        ind+=1
                    else:
                        continue
                else:
                    continue
        
        #Verification de l'inclusion de la symmétrie d'inversion
        invsym_bool = IsInvSymIn(SymRec, nsym)
        if invsym_bool==False:
            for i in range(len(vec_q_toind)):
                q=ind_q_tovec[i]
                if (-q[0], -q[1], -q[2]) not in vec_q_toind.keys():
                    vec_q_toind[(-q[0], -q[1], -q[2])] = ind
                    ind_q_tovec[ind] = (-q[0], -q[1], -q[2])
                    qsym = (symdict[i][0], symdict[i][1])
                    ind_q_to_vec[ind] = np.multiply(-q, nk)
                    sym_dict[ind] = (nsym+1, i, qsym)
                    ind += 1
                else:
                    continue
        nq = len(sym_dict)

        #Liste des vecteurs G
        G_dict = {}
        ind_G_to_vec = {}
        for i in range(ng):
            G_dict[(G[i, 0], G[i, 1], G[i, 2])] = i
            ind_G_to_vec[i] = np.multiply([G[i, 0], G[i, 1], G[i, 2]], nk)
        #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau)
        qibzG_dict = {}
        ind_qibzG_dict = {}
        qbzG_dict = {}
        for i in range(nkpt):
            for j in range(ng):
                qG = kpoints[i].frac_coords+G[j]
                qibzG_dict[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qibzG_dict[j+i*ng] = (qG[0], qG[1], qG[2])

        for i in range(nq):
            for j in range(ng):
                q = ind_q_tovec[i]
                qG = q+G[j]
                qbzG_dict[(qG[0], qG[1], qG[2])] = j+i*ng
        
        #Test des vecteurs dans le sets après rotation
        vec_qG_to_ind_inset, ind_qG_to_vec_inset = {},{}
        count=0
        for i in range(len(qibzG_dict)):
            qibzG = ind_qibzG_dict[i]
            in_qGset = True
            for j in range(nsym):
                qGrot = np.matmul(SymRec[j], [qibzG[0], qibzG[1], qibzG[2]])
                qGtest = (qGrot[0], qGrot[1], qGrot[2])
                if qGtest not in qbzG_dict.keys():
                    in_qGset = False
                    break
            if in_qGset == True:
                vec_qG_to_ind_inset[qibzG] = count
                ind_qG_to_vec_inset[count] = qibzG
                count+=1

        #Liste des veteurs q+G valables
        nk = fsk(kpoints)
        vec_qG_to_ind_without_border, ind_qG_to_vec_without_border = {}, {}
        vec_to_ind_to_pass, ind_to_vec_to_pass = {}, {}
        secondcount = 0
        for i in range(count):
            qibzG = ind_qG_to_vec_inset[i]
            for j in range(nsym):
                qG = np.matmul(SymRec[j], [qibzG[0], qibzG[1], qibzG[2]])
                if (qG[0], qG[1], qG[2]) not in vec_qG_to_ind_without_border.keys():
                    qGind = np.round(np.multiply([qG[0], qG[1], qG[2]], nk))
                    vec_to_ind_to_pass[(qGind[0], qGind[1], qGind[2])] = secondcount
                    ind_to_vec_to_pass[secondcount] = (qGind[0], qGind[1], qGind[2])
                    vec_qG_to_ind_without_border[(qG[0], qG[1], qG[2])] = secondcount
                    ind_qG_to_vec_without_border[secondcount] = (qG[0], qG[1], qG[2])
                    secondcount += 1

        nvec = len(vec_qG_to_ind_without_border) 
        vec_table_without_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:, 1])), np.amax(np.abs(vec_table_without_border[:, 2]))
        n1, n2, n3=(2*s1)*nk[0]+1, (2*s2)*nk[1]+1, (2*s3)*nk[2]+1

        chi0GG = np.zeros((nq, ng, ng), dtype = complex)
        ic = complex(0, 1)
        for i in range(nq):
            q = ind_q_tovec[i]
            qvec = [q[0], q[1], q[2]]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                qorigin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
                t = [0, 0, 0]
            else:
                SymR = SymRec[SymData[0]]
                t = [0, 0, 0]
                qorigin = kpoints[SymData[1]]
            chi0 = sus_ncfile.reader.read_wggmat(qorigin).wggmat
            for j in range(ng):
                G1 = G[j]
                qG1vec = qvec+G1
                qG1 = (qG1vec[0], qG1vec[1], qG1vec[2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                SG1 = np.matmul(np.linalg.inv(SymR), G1)
                indchi1 = G_dict[(SG1[0], SG1[1], SG1[2])]
                for k in range(ng):
                    G2 = G[k]
                    qG2vec = qvec+G2
                    qG2 = (qG2vec[0], qG2vec[1], qG2vec[2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        ind2 = vec_qG_to_ind_without_border[qG2]
                        SG2 = np.matmul(np.linalg.inv(SymR), G2)
                        indchi2 = G_dict[(SG2[0], SG2[1], SG2[2])]
                        chi0GG[i, j, k] = chi0[omega, indchi1, indchi2]#*cmath.exp(ic*np.dot(t, G2-G1))


        return chi0GG, vec_to_ind_to_pass, ind_to_vec_to_pass, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G


    elif opt=='FullBZ1':
        #total number of vector possible
        nvec=ng*nkpt
        #Small code to get the size of the sampling grid used for the BZ
        kpoint_tab=kpoints.frac_coords
        ind1,ind2,ind3=kpoint_tab[:,0]!=0,kpoint_tab[:,1]!=0,kpoint_tab[:,2]!=0
        kpoints_tab1,kpoints_tab2,kpoints_tab3=kpoint_tab[:,0][ind1],kpoint_tab[:,1][ind2],kpoint_tab[:,2][ind3]
        mk1,mk2,mk3=np.amin(np.abs(kpoints_tab1)),np.amin(np.abs(kpoints_tab2)),np.amin(np.abs(kpoints_tab3))
        nk=[np.round(1/mk1),np.round(1/mk2),np.round(1/mk3)]
        #Initialisation des dictionnaires de vecteurs:
        vec_qG_to_ind,ind_qG_to_vec={},{}
        vec_table=np.zeros((nvec,3),dtype=int)
        for i in range(nkpt):
            for j in range(ng):
                ind=j+i*ng
                vec_table[ind]=np.round(np.multiply((kpoints[i].frac_coords+G[j]),nk))
                qG=(vec_table[ind,0],vec_table[ind,1],vec_table[ind,2])
                vec_qG_to_ind[qG]=ind
                ind_qG_to_vec[ind]=qG
        #Creation d'un second dictionnaire sans les points frontières :
        vec_qG_to_ind_without_border={}
        ind_qG_to_vec_without_border={}
        #Filtre des vecteur pour garder uniquement les vecteurs internes 
        count=0 #Numbers of inner vectors
        for i in range(nvec):
            qG=ind_qG_to_vec[i]
            qGopp=(-qG[0],-qG[1],-qG[2])
            if qGopp not in vec_qG_to_ind.keys():
                continue
            else:
                vec_qG_to_ind_without_border[qG]=count
                ind_qG_to_vec_without_border[count]=qG
                count+=1
        #print(vec_qG_to_ind_without_border)
        vec_table_without_border=np.zeros((count,3),dtype=int)
        for i in range(count):
            qG=ind_qG_to_vec_without_border[i]
            vec_table_without_border[i]=[qG[0],qG[1],qG[2]]
        s1,s2,s3=np.amax(np.abs(vec_table_without_border[:,0])),np.amax(np.abs(vec_table_without_border[:,1])),np.amax(np.abs(vec_table_without_border[:,2]))
        n1,n2,n3=2*s1+1,2*s2+1,2*s3+1
        #print(len(vec_qG_to_ind_without_border))
        chi0GG=np.zeros((count,count),dtype=complex)
        

        for i in range(nkpt):
            chi0=sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat
            for j in range(ng):
                ind1=j+i*ng
                qG1=(vec_table[ind1,0],vec_table[ind1,1],vec_table[ind1,2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                indvec1=vec_qG_to_ind_without_border[qG1]
                for k in range(ng):
                    ind2=k+i*ng
                    qG2=(vec_table[ind2,0],vec_table[ind2,1],vec_table[ind2,2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        indvec2=vec_qG_to_ind_without_border[qG2]
                        chi0GG[indvec1,indvec2]=chi0[omega,j,k]
        
        return chi0GG,vec_qG_to_ind_without_border,ind_qG_to_vec_without_border,n1,n2,n3


                
    else:
        return str(opt)+' is not a valid option, read the documentation to see the list of options'


def FFT_chi0(filename, opt1 = "FullBZ", opt2 = "Standard", omega = 0):
    
    if opt2 == "Standard":
        chi0GG, vec_qG_to_ind, ind_qG_to_vec, n1, n2, n3 = Build_Chi0GG2D(filename, opt1, omega)
        nqG1, nqG2 = chi0GG.shape
        n1, n2, n3=round(n1), round(n2), round(n3)

        #Première FFT:
        fftboxsize = round(n1*n2*n3)
        chi0rG = np.zeros((fftboxsize, nqG2), dtype = complex)
        print("Starting first FFT")
        for i in range(nqG2):
            chi0GG2 = chi0GG[:, i]
            FFTBox = np.zeros((n1, n2, n3), dtype = complex)
            for j in range(nqG1):
                qG = ind_qG_to_vec[j]
                FFTBox[round(qG[0]), round(qG[1]), round(qG[2])] = chi0GG2[j]
            FFT = np.fft.ifftn(FFTBox)
            chi0rG[:, i]= np.reshape(FFT, fftboxsize)
    
        #Seconde FFT:
        chi0rr = np.zeros((fftboxsize, n1, n2, n3), dtype = complex)
        print("Starting second FFT")
        for i in range(fftboxsize):
            chi0r1G = chi0rG[i, :]
            FFTBox = np.zeros((n1, n2, n3), dtype = complex)
            for j in range(nqG2):
                qG = ind_qG_to_vec[j]
                FFTBox[-round(qG[0]), -round(qG[1]), -round(qG[2])] = chi0r1G[j]
            FFT = np.fft.ifftn(FFTBox)  
            chi0rr[i] = FFT
        chi0rr_out0 = np.reshape(chi0rr, (n1, n2, n3, n1, n2, n3))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        return chi0rr_out 


    elif opt2 == "Kaltak":
        chi0GG, vec_qG_to_ind, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G = Build_Chi0GG(filename, opt1, omega)
        nq, ng1, ng2 = chi0GG.shape
        if ng1 != ng2:
            return "There is a problem in the code"
        ng=ng1
        nqg = nq * ng
        n1, n2, n3=round(n1), round(n2), round(n3)
        fftboxsize = round(n1*n2*n3)
        maxG1,maxG2,maxG3=np.amax(np.abs(G[:,0])),np.amax(np.abs(G[:,1])),np.amax(np.abs(G[:,2]))
        n4, n5, n6 = int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
        fftboxsizeG = n4*n5*n6
        # Première FFT
        print("Starting first FFT")
        chi0rG = np.zeros((nq, fftboxsizeG, ng), dtype = complex)
        qG = np.zeros((nqg, 3), dtype = int)
        for i in range(nq):
            for j in range(ng):
                chi0GqG2 = chi0GG[i, :, j]
                FFTBox = np.zeros((n4, n5, n6), dtype = complex)
                for k in range(ng):
                    FFTBox[round(G[k,0]), round(G[k,1]), round(G[k,2])] = chi0GqG2[k]
                FFT = np.fft.ifftn(FFTBox)
                chi0rG[i, :, j]= np.reshape(FFT, fftboxsizeG)

        # Seconde FFT
        print("Starting second FFT")
        chi0rr = np.zeros((fftboxsizeG, n1, n2, n3), dtype = complex)
        for i in range(fftboxsizeG):
            FFTBox = np.zeros((n1, n2, n3), dtype = complex)
            for j in range(nq):
                q = ind_q_to_vec[j]
                for k in range(ng):
                    G2 = ind_G_to_vec[k]
                    qG2 = q+G2
                    FFTBox[-round(qG2[0]), -round(qG2[1]), -round(qG2[2])] = chi0rG[j, i, k]
            FFT = np.fft.ifftn(FFTBox)  
            chi0rr[i, :, :, :]=FFT
        chi0rr_out0 = np.reshape(chi0rr, (n4, n5, n6, n1, n2, n3))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        return chi0rr_out
    
    

        
    else:
        return "The second option is not valid, see the function definition to know the different possibilities"


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
    


def Vis_tool_Bulk(chi0rr, R, A, B, C, nk, N=10000, isomin=2e-9):
    
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

def Vis_tool(chi0rr, R, A, B, C, nk, nat_cell, pos_red, N=10000, isomin=2e-9):
    
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



def Build_Chi0GG2D(filename, opt, omega = 0):
    sus_ncfile, kpoints, nw, ng, nkpt, G, nqg = openfile(filename)
    if opt == 'FullBZ':
        nvec = nkpt*ng
        nk = fsk(kpoints)
        #Creation d'un dictionnaire :
        vec_qG_to_ind = {}
        ind_qG_to_vec = {}
        vec_table = np.zeros((nvec,3), dtype = int)

        for i in range(nkpt):
            for j in range(ng):
                ind = j + i * ng
                vec_table[ind] = np.round(np.multiply((kpoints[i].frac_coords + G[j]), nk))
                qG = (vec_table[ind,0], vec_table[ind,1], vec_table[ind,2])
                vec_qG_to_ind[qG] = ind 
                ind_qG_to_vec[ind] = qG

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

        vec_table_without_border = np.zeros((count, 3), dtype = int)
        for i in range(count):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:,1])), np.amax(np.abs(vec_table_without_border[:,2]))
        n1, n2, n3 = 2*s1+1, 2*s2+1, 2*s3+1

        #Initialisation of chi0 :
        chi0GG = np.zeros((count, count), dtype = complex)

        for i in range(nkpt):
            chi0=sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat
            for j in range(ng):
                ind1 = j+i*ng
                qG1 = (vec_table[ind1, 0], vec_table[ind1, 1], vec_table[ind1, 2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                indvec1 = vec_qG_to_ind_without_border[qG1]
                for k in range(ng):
                    ind2 = k+i*ng
                    qG2 = (vec_table[ind2, 0], vec_table[ind2, 1], vec_table[ind2, 2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        indvec2 = vec_qG_to_ind_without_border[qG2]
                        chi0GG[indvec1, indvec2] = chi0[omega, j, k]
        """ for i in range(count):
            for j in range(count):
                vec_sym_1 = ind_qG_to_vec_without_border[i]
                vec_sym_2 = ind_qG_to_vec_without_border[j]
                k, l = vec_qG_to_ind_without_border[(-vec_sym_1[0], -vec_sym_1[1], -vec_sym_1[2])], vec_qG_to_ind_without_border[(-vec_sym_2[0], -vec_sym_2[1], -vec_sym_2[2])]
                chi0GG[i, j] = 1/2*(chi0GG[i, j]+chi0GG[l, k])
                chi0GG[l, k]=chi0GG[i, j] """

        return chi0GG, vec_qG_to_ind_without_border, ind_qG_to_vec_without_border, n1, n2, n3
    
    elif opt=='FromSym':
        
        structure = abipy.core.structure.Structure.from_file(filename)
        Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
        SymRec = Sym.symrec
        TRec = Sym.tnons
        Tim_Rev = Sym.has_timerev
        nsym = len(SymRec)

        #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ

        vec_q_toind, ind_q_tovec, sym_dict = {},{},{}
        ind = 0
        for i in range(nsym):
            for j in range(nkpt):
                q = np.round(np.matmul(SymRec[i], kpoints[j].frac_coords),5)#+TRec[i]
                if (q[0], q[1], q[2]) not in vec_q_toind.keys():
                    if np.amax(q) <= 0.5 and np.amin(q) >- 0.5:
                        vec_q_toind[(q[0], q[1], q[2])] = ind
                        ind_q_tovec[ind] = (q[0], q[1], q[2])
                        sym_dict[ind]=(i, j, (0, 0))
                        ind+=1
                    else:
                        continue
                else:
                    continue
        
        #Verification de l'inclusion de la symmétrie d'inversion
        invsym_bool = IsInvSymIn(SymRec, nsym)
        
        if invsym_bool==False:
            for i in range(len(vec_q_toind)):
                q=ind_q_tovec[i]
                if (-q[0], -q[1], -q[2]) not in vec_q_toind.keys():
                    vec_q_toind[(-q[0], -q[1], -q[2])] = ind
                    ind_q_tovec[ind] = (-q[0], -q[1], -q[2])
                    qsym = (symdict[i][0], symdict[i][1])
                    sym_dict[ind] = (nsym+1, i, qsym)
                    ind += 1
                else:
                    continue
        nq = len(sym_dict)
        print(vec_q_toind)
        #Liste des vecteurs G
        G_dict = {}
        for i in range(ng):
            G_dict[(G[i, 0], G[i, 1], G[i, 2])] = i

        #Listes des vecteurs qibz+G et qbz+G (dictionnaire + tableau)
        qibzG_dict = {}
        ind_qibzG_dict = {}
        qbzG_dict = {}
        for i in range(nkpt):
            for j in range(ng):
                qG = kpoints[i].frac_coords+G[j]
                qibzG_dict[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qibzG_dict[j+i*ng] = (qG[0], qG[1], qG[2])

        for i in range(nq):
            for j in range(ng):
                q = ind_q_tovec[i]
                qG = q+G[j]
                qbzG_dict[(qG[0], qG[1], qG[2])] = j+i*ng
        
        #Test des vecteurs dans le sets après rotation
        vec_qG_to_ind_inset, ind_qG_to_vec_inset = {},{}
        count=0
        for i in range(len(qibzG_dict)):
            qibzG = ind_qibzG_dict[i]
            in_qGset = True
            for j in range(nsym):
                qGrot = np.matmul(SymRec[j], [qibzG[0], qibzG[1], qibzG[2]])
                qGtest = (qGrot[0], qGrot[1], qGrot[2])
                if qGtest not in qbzG_dict.keys():
                    in_qGset = False
                    break
            if in_qGset == True:
                vec_qG_to_ind_inset[qibzG] = count
                ind_qG_to_vec_inset[count] = qibzG
                count+=1

        #Liste des veteurs q+G valables
        nk = fsk(kpoints)
        vec_qG_to_ind_without_border, ind_qG_to_vec_without_border = {}, {}
        vec_to_ind_to_pass, ind_to_vec_to_pass = {}, {}
        secondcount = 0
        for i in range(count):
            qibzG = ind_qG_to_vec_inset[i]
            for j in range(nsym):
                qG = np.matmul(SymRec[j], [qibzG[0], qibzG[1], qibzG[2]])
                if (qG[0], qG[1], qG[2]) not in vec_qG_to_ind_without_border.keys():
                    qGind = np.round(np.multiply([qG[0], qG[1], qG[2]], nk))
                    vec_to_ind_to_pass[(qGind[0], qGind[1], qGind[2])] = secondcount
                    ind_to_vec_to_pass[secondcount] = (qGind[0], qGind[1], qGind[2])
                    vec_qG_to_ind_without_border[(qG[0], qG[1], qG[2])] = secondcount
                    ind_qG_to_vec_without_border[secondcount] = (qG[0], qG[1], qG[2])
                    secondcount += 1

        nvec = len(vec_qG_to_ind_without_border) 
        vec_table_without_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qG_to_vec_without_border[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:, 1])), np.amax(np.abs(vec_table_without_border[:, 2]))
        n1, n2, n3=(2*s1)*nk[0]+1, (2*s2)*nk[1]+1, (2*s3)*nk[2]+1

        chi0GG = np.zeros((nvec, nvec), dtype=complex)
        ic = complex(0, 1)
        for i in range(nq):
            q = ind_q_tovec[i]
            qvec = [q[0], q[1], q[2]]
            SymData = sym_dict[i]
            if SymData[0] == nsym+1:
                SymR1 = [[-1,0,0], [0,-1,0], [-1,0,0]]
                SymR2 = SymRec[SymData[2][0]]
                SymR = np.matmul(SymR1, SymR2)
                qorigin = SymRec[2][1]
                #t=TRec[SymData[2][0]]
                #t = [0, 0, 0]
            else:
                SymR = SymRec[SymData[0]]
                #t = [0, 0, 0]
                qorigin = kpoints[SymData[1]]
            chi0 = sus_ncfile.reader.read_wggmat(qorigin).wggmat
            for j in range(ng):
                G1 = G[j]
                qG1vec = qvec+G1
                qG1 = (qG1vec[0], qG1vec[1], qG1vec[2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                ind1 = vec_qG_to_ind_without_border[qG1]
                SG1 = np.matmul(np.linalg.inv(SymR), G1)
                indchi1 = G_dict[(SG1[0], SG1[1], SG1[2])]
                for k in range(ng):
                    G2 = G[k]
                    qG2vec = qvec+G2
                    qG2 = (qG2vec[0], qG2vec[1], qG2vec[2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        ind2 = vec_qG_to_ind_without_border[qG2]
                        SG2 = np.matmul(np.linalg.inv(SymR), G2)
                        indchi2 = G_dict[(SG2[0], SG2[1], SG2[2])]
                        chi0GG[ind1, ind2] = chi0[omega, indchi1, indchi2]#*cmath.exp(ic*np.dot(t, G2-G1))

        return chi0GG, vec_to_ind_to_pass, ind_to_vec_to_pass, n1, n2, n3


    elif opt=='FullBZ1':
        #total number of vector possible
        nvec=ng*nkpt
        #Small code to get the size of the sampling grid used for the BZ
        kpoint_tab=kpoints.frac_coords
        ind1,ind2,ind3=kpoint_tab[:,0]!=0,kpoint_tab[:,1]!=0,kpoint_tab[:,2]!=0
        kpoints_tab1,kpoints_tab2,kpoints_tab3=kpoint_tab[:,0][ind1],kpoint_tab[:,1][ind2],kpoint_tab[:,2][ind3]
        mk1,mk2,mk3=np.amin(np.abs(kpoints_tab1)),np.amin(np.abs(kpoints_tab2)),np.amin(np.abs(kpoints_tab3))
        nk=[np.round(1/mk1),np.round(1/mk2),np.round(1/mk3)]
        #Initialisation des dictionnaires de vecteurs:
        vec_qG_to_ind,ind_qG_to_vec={},{}
        vec_table=np.zeros((nvec,3),dtype=int)
        for i in range(nkpt):
            for j in range(ng):
                ind=j+i*ng
                vec_table[ind]=np.round(np.multiply((kpoints[i].frac_coords+G[j]),nk))
                qG=(vec_table[ind,0],vec_table[ind,1],vec_table[ind,2])
                vec_qG_to_ind[qG]=ind
                ind_qG_to_vec[ind]=qG
        #Creation d'un second dictionnaire sans les points frontières :
        vec_qG_to_ind_without_border={}
        ind_qG_to_vec_without_border={}
        #Filtre des vecteur pour garder uniquement les vecteurs internes 
        count=0 #Numbers of inner vectors
        for i in range(nvec):
            qG=ind_qG_to_vec[i]
            qGopp=(-qG[0],-qG[1],-qG[2])
            if qGopp not in vec_qG_to_ind.keys():
                continue
            else:
                vec_qG_to_ind_without_border[qG]=count
                ind_qG_to_vec_without_border[count]=qG
                count+=1
        #print(vec_qG_to_ind_without_border)
        vec_table_without_border=np.zeros((count,3),dtype=int)
        for i in range(count):
            qG=ind_qG_to_vec_without_border[i]
            vec_table_without_border[i]=[qG[0],qG[1],qG[2]]
        s1,s2,s3=np.amax(np.abs(vec_table_without_border[:,0])),np.amax(np.abs(vec_table_without_border[:,1])),np.amax(np.abs(vec_table_without_border[:,2]))
        n1,n2,n3=2*s1+1,2*s2+1,2*s3+1
        #print(len(vec_qG_to_ind_without_border))
        chi0GG=np.zeros((count,count),dtype=complex)
        

        for i in range(nkpt):
            chi0=sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat
            for j in range(ng):
                ind1=j+i*ng
                qG1=(vec_table[ind1,0],vec_table[ind1,1],vec_table[ind1,2])
                if qG1 not in vec_qG_to_ind_without_border.keys():
                    continue
                indvec1=vec_qG_to_ind_without_border[qG1]
                for k in range(ng):
                    ind2=k+i*ng
                    qG2=(vec_table[ind2,0],vec_table[ind2,1],vec_table[ind2,2])
                    if qG2 not in vec_qG_to_ind_without_border.keys():
                        continue
                    else:
                        indvec2=vec_qG_to_ind_without_border[qG2]
                        chi0GG[indvec1,indvec2]=chi0[omega,j,k]
        
        return chi0GG,vec_qG_to_ind_without_border,ind_qG_to_vec_without_border,n1,n2,n3


                
    else:
        return str(opt)+' is not a valid option, read the documentation to see the list of options'
    

def FFT_chi02D(filename, opt1, omega=0):
    
    chi0GG, vec_qG_to_ind, ind_qG_to_vec, n1, n2, n3 = Build_Chi0GG2D(filename, opt1, omega)
    nqG1, nqG2 = chi0GG.shape
    n1, n2, n3=round(n1), round(n2), round(n3)

    #Première FFT:
    fftboxsize = round(n1*n2*n3)
    chi0rG = np.zeros((fftboxsize, nqG2), dtype = complex)

    for i in range(nqG2):
        chi0GG2 = chi0GG[:, i]
        FFTBox = np.zeros((n1, n2, n3), dtype = complex)
        for j in range(nqG1):
            qG = ind_qG_to_vec[j]
            FFTBox[round(qG[0]), round(qG[1]), round(qG[2])] = chi0GG2[j]
        FFT = np.fft.ifftn(FFTBox)
        chi0rG[:, i]= np.reshape(FFT, fftboxsize)
    
    #Seconde FFT:
    chi0rr = np.zeros((fftboxsize, n1, n2, n3), dtype = complex)

    for i in range(fftboxsize):
        chi0r1G = chi0rG[i, :]
        FFTBox = np.zeros((n1, n2, n3), dtype = complex)
        for j in range(nqG2):
            qG = ind_qG_to_vec[j]
            FFTBox[-round(qG[0]), -round(qG[1]), -round(qG[2])] = chi0r1G[j]
        FFT = np.fft.ifftn(FFTBox)  
        chi0rr[i] = FFT
    chi0rr_out = np.reshape(chi0rr, (n1, n2, n3, n1, n2, n3))
    return chi0rr_out
    
