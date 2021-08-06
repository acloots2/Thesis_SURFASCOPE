######Xavier's Algorithm#########
from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
import abipy
import numpy as np
import cmath
import math
import pointcloud as pc
from scipy.interpolate import RegularGridInterpolator
import plotly.graph_objects as go
import time

def charge_nearby(chi0rr, dV, dr):
    n1, n2, n3, n4, n5, n6 = chi0rr.shape
    charge = 0

    for i in range(dV[0]-dr, dV[0]+dr):
        if i >= n4:
            i = i - n4
        for j in range(dV[1]-dr, dV[1]+dr):
            if j >= n5:
                j = i - n5
            for k in range(dV[2]-dr, dV[2]+dr):
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

def Build_Chi0GG(filename, opt, omega = 0):
    sus_ncfile, kpoints, ng, nkpt, G = openfile(filename)
    if opt == 'FullBZ':
        nvec = nkpt*ng
        print(nkpt, ng, nvec)
        nk = fsk(kpoints)
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
    
    elif opt=='FromSym':
        
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

        """ nvec = nq*ng
        vec_table = np.zeros((nvec,3), dtype = int) """
        for i in range(nkpt):
            for j in range(ng):
                kpt = kpoints[i].frac_coords
                G_vec = G[j]
                qG = np.round(np.multiply([kpt[0]+G_vec[0], kpt[1]+G_vec[1], kpt[2]+G_vec[2]], nk))
                vec_qibzG_to_ind[(qG[0], qG[1], qG[2])] = j+i*ng
                ind_qibzG_to_vec[j+i*ng] = (qG[0], qG[1], qG[2])
                """ ind = j + i * ng
                vec_table[ind] = np.round(np.multiply((kpoints[i].frac_coords + G[j]), nk)) """
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


        return chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk
    elif opt == "FromSymTest":
        start_time = time.time()
        structure = abipy.core.structure.Structure.from_file(filename)
        Sym = abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
        dict_struct = structure.to_abivars()
        natom = dict_struct["natom"]
        SymRec = Sym.symrec
        print(len(SymRec))
        #Trec = Sym.tnons
        nsym = len(SymRec)
        #nsym = round(len(SymRec)/natom)
        nk = fsk(kpoints)
        #Obtenir tout les q-points dans la BZ depuis ceux dans l'IBZ. Needs to pay attention to rounding errors+needs to use Umklapp vectors to get all the data
        #print(Trec)
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
        print(len(ind_q_to_vec))
        print(ind_q_to_vec)
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
        print(ng)
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
                qG_rot = np.round(np.matmul(SymRec[j],[qibzG_vec[0], qibzG_vec[1], qibzG_vec[2]]))
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
        vec_table_without_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qbzG_to_vec[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]
        
        #Settings for the building of chi0GG
               
        #print("Dict of symmetry initialized")

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:, 1])), np.amax(np.abs(vec_table_without_border[:, 2]))
        n1, n2, n3=(2*s1)+1, (2*s2)+1, (2*s3)+1

        elapsed_1 = time.time()-start_time
        print("the initialization of the dictionnaries and gathering of the information about the valid vectors took", elapsed_1, "seconds")
        chi0GG = np.zeros((nq, ng, ng), dtype = complex)
        smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)

        for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega]

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
        print("Sym ok")
        #print(vec_from_sym)
        count = 0
        ind_pairqG_to_pair = {}
        #item_to_loc = {}
        
        for q_vec in vec_q_to_ind.keys():
            print(q_vec)
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
            ind_sm_chi011,  ind_sm_chi012= ind_qibzG1%ng, ind_qibzG2%ng
            ind_sm_chi021,  ind_sm_chi022= round((ind_qibzG1-ind_sm_chi011)/ng), round((ind_qibzG2-ind_sm_chi012)/ng)
            if ind_sm_chi021 != ind_sm_chi022:
                print("Strange")
            chi0GG[i, j, k] = smallchi0GG[ind_sm_chi021, ind_sm_chi011,  ind_sm_chi012]#*cmath.exp(ic*np.dot(t, G2_vec-G1_vec))
        elapsed_3 = time.time()-(elapsed_2+elapsed_1+start_time)
        print("the building of chi0GG took ", elapsed_3, "seconds")
        return chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk
    elif opt == "FromSym2":
        start_time = time.time()
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
                qG_rot = np.round(np.matmul(SymRec[j],[qibzG_vec[0], qibzG_vec[1], qibzG_vec[2]]))
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
        vec_table_without_border = np.zeros((nvec, 3), dtype=int)
        for i in range(nvec):
            qG = ind_qbzG_to_vec[i]
            vec_table_without_border[i] = [qG[0], qG[1], qG[2]]
        
        #Settings for the building of chi0GG
               
        print("Dict of symmetry initialized")

        s1, s2, s3 = np.amax(np.abs(vec_table_without_border[:, 0])), np.amax(np.abs(vec_table_without_border[:, 1])), np.amax(np.abs(vec_table_without_border[:, 2]))
        n1, n2, n3=(2*s1)+1, (2*s2)+1, (2*s3)+1

        elapsed_1 = time.time()-start_time
        print("the initialization of the dictionnaries and gathering of the information about the valid vectors took", elapsed_1, "seconds")
        chi0GG = np.zeros((nq, ng, ng), dtype = complex)
        smallchi0GG = np.zeros((nkpt, ng, ng), dtype = complex)

        for i in range(nkpt):
            q_origin = kpoints[i]
            chi0 = sus_ncfile.reader.read_wggmat(q_origin).wggmat
            smallchi0GG[i] = chi0[omega]

        for i in range(nkpt):
            q_vec = ind_q_to_vec[i]
            for j in range(ng):
                G_vec = ind_G_to_vec[j]
                if (q_vec[0]+G_vec[0], q_vec[1]+G_vec[1], q_vec[2]+G_vec[2]) in vec_with_missing_sym.keys():
                    smallchi0GG[i, j, :] = np.zeros(ng)
                    smallchi0GG[i, :, j] = np.zeros(ng)
        
        for i in range(nkpt):
            q_vec = ind_q_to_vec[i]
            for j in range(ng):
                G1_vec = ind_G_to_vec[j]
                for k in range(ng):
                    G2_vec = ind_G_to_vec[k]
                    chi0 = smallchi0GG[i, j, k]
                    if chi0 == 0:
                        continue
                    else:
                        for l in range(nsym):
                            SqG1 = np.round(np.matmul(SymRec[l],[q_vec[0]+G1_vec[0], q_vec[1]+G1_vec[1], q_vec[2]+G1_vec[2]]))
                            Sq_vec = [SqG1[0]%nk[0], SqG1[1]%nk[1], SqG1[2]%nk[2]]
                            if Sq_vec[0] > np.round(0.5*nk[0]):
                                Sq_vec[0] = Sq_vec[0]-nk[0]
                            if Sq_vec[1] > np.round(0.5*nk[1]):
                                Sq_vec[1] = Sq_vec[1]-nk[1]
                            if Sq_vec[2] > np.round(0.5*nk[2]):
                                Sq_vec[2] = Sq_vec[2]-nk[2]
                            o = vec_q_to_ind[(Sq_vec[0], Sq_vec[1], Sq_vec[2])]
                            p = vec_G_to_ind[(SqG1[0]-Sq_vec[0], SqG1[1]-Sq_vec[1], SqG1[2]-Sq_vec[2])]
                            SqG2 = np.round(np.matmul(SymRec[l],[q_vec[0]+G2_vec[0], q_vec[1]+G2_vec[1], q_vec[2]+G2_vec[2]]))
                            if (SqG2[0], SqG2[1], SqG2[2]) not in vec_qbzG_to_ind.keys():
                                print("Strange1")
                            Sq2_vec = [SqG2[0]%nk[0], SqG2[1]%nk[1], SqG2[2]%nk[2]]
                            if Sq2_vec[0] > np.round(0.5*nk[0]):
                                Sq2_vec[0] = Sq2_vec[0]-nk[0]
                            if Sq2_vec[1] > np.round(0.5*nk[1]):
                                Sq2_vec[1] = Sq2_vec[1]-nk[1]
                            if Sq2_vec[2] > np.round(0.5*nk[2]):
                                Sq2_vec[2] = Sq2_vec[2]-nk[2]
                            q = vec_q_to_ind[(Sq2_vec[0], Sq2_vec[1], Sq2_vec[2])]
                            r = vec_G_to_ind[(SqG2[0]-Sq2_vec[0], SqG2[1]-Sq2_vec[1], SqG2[2]-Sq2_vec[2])]
                            if o != q:
                                print("Strange2")
                            chi0GG[o, p, r] = smallchi0GG[i, j, k]
        return chi0GG, ind_qbzG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk
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
        n4, n5, n6 = int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
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
        chi0rr_out0 = np.reshape(chi0rr, (n4_fft, n5_fft, n6_fft, n1_fft, n2_fft, n3_fft))
        chi0rr_out = chi0rr_out0 * chi0rr_out0.size
        elapsed2 = time.time()-start_time-elapsed1
        print("The second FFT took ", elapsed2, " seconds")
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