import numpy as np
import math 
import cmath
import abipy
from abipy.electrons.scr import ScrFile
import time
import matplotlib.pyplot as plt

def openfile(filename):
    #Get the main data
    ##Open the file
    sus_ncfile = ScrFile(filename)
    ##Get the list of kpoints (red coord)
    kpoints=sus_ncfile.reader.kpoints 
    ##Get the number of G-vectors
    ng=sus_ncfile.reader.ng 
    ##Get the number of kpoints
    nkpt=len(kpoints)
    ##Get the chi0 matrix of q=0 0 0 (needed to get the G sphere)
    chi0=sus_ncfile.reader.read_wggmat(kpoints[0])
    ##Get the list of G vectors (red coord)
    G=chi0.gsphere.gvecs
    return sus_ncfile, kpoints, ng, nkpt, G


#returns an np.array of dimension (nqgx3) with all the q+G vectors
#Note that the vectors are scaled such that each coordinate is an integer (smallest possible)
def qG_Grid(nqg,kpoints,nkpt,ng,G):
    Grid=np.zeros((nqg,3),dtype=int)
    nk=FindSmallerKpoint(kpoints)
    for i in range(nkpt):
        for j in range(ng):
            Grid[nkpt*j+i]=np.round(np.multiply((kpoints[i].frac_coords+G[j]),nk))
    return Grid


def qG_Grid_sym(nqg,kpoints,nkpt,ng,G):
    Grid=np.zeros((nqg,3),dtype=int)
    mk1,mk2,mk3=FindSmallerKpoint(kpoints)
    nk=[round(1/mk1),round(1/mk2),round(1/mk3)]
    for i in range(nkpt):
        for j in range(ng):
            Grid[nkpt*j+i]=np.round(np.multiply((kpoints[i]+G[j]),nk))
    return Grid


#returns a dictionnary of linking the index in the matrix to the q+G vector (indtovec) and the opposite (vectoind)
#It completes the dictionnary such that each q+G and -q-G are present
def Dict_init(Grid,nqg):
    vectoind={}
    indtovec={}
    newind=nqg
    for i in range(nqg):
        vectoind[(Grid[i,0],Grid[i,1],Grid[i,2])]=i
        indtovec[i]=(Grid[i,0],Grid[i,1],Grid[i,2]) 
    for i in range(nqg):
        vec=indtovec[i]
        vectest=(-vec[0],-vec[1],-vec[2])
        if vectest not in vectoind.keys():
            vectoind[vectest]=newind
            indtovec[newind]=vectest
            newind+=1
    return vectoind,indtovec


#Return the smallest step for the coordinates of the q vectors (reduced coordinates)
def FindSmallerKpoint(kpoints):
    kpoint_tab=kpoints.frac_coords
    ind1=kpoint_tab[:,0]!=0
    ind2=kpoint_tab[:,1]!=0
    ind3=kpoint_tab[:,2]!=0
    kpoints_tab1=kpoint_tab[:,0][ind1]
    kpoints_tab2=kpoint_tab[:,1][ind2]
    kpoints_tab3=kpoint_tab[:,2][ind3]
    #print(kpoints_tab3)
    A,B,C=True,True,True
    if kpoints_tab1.size==0:
        mk1=0
        nk1=1
        A=False
    if kpoints_tab2.size==0:
        mk2=0
        nk1=1
        B=False
    if kpoints_tab3.size==0:
        mk3=0
        nk3=1
        C=False
    if A==True and B==True and C==True:
        mk1,mk2,mk3=np.min(np.abs(kpoints_tab1)),np.min(np.abs(kpoints_tab2)),np.min(np.abs(kpoints_tab3))
        nk=[np.round(1/mk1),np.round(1/mk2),np.round(1/mk3)]
    elif A==True and B==True and C==False:
        mk1,mk2=np.min(np.abs(kpoints_tab1)),np.min(np.abs(kpoints_tab2))
        nk=[np.round(1/mk1),np.round(1/mk2),nk3]
    elif A==True and C==True and B==False:
        mk1,mk3=np.min(np.abs(kpoints_tab1)),np.min(np.abs(kpoints_tab3))
        nk=[np.round(1/mk1),nk2,np.round(1/mk3)]
    elif B==True and C==True and B==False:
        mk2,mk3=np.min(np.abs(kpoints_tab2)),np.min(np.abs(kpoints_tab3))
        nk=[nk1,np.round(1/mk2),np.round(1/mk3)]
    elif A==True and B==False and C==False:
        mk1=np.min(np.abs(kpoints_tab1))
        nk=[np.round(1/mk1),nk2,nk3]
    elif A==False and B==True and C==False:
        mk2=np.min(np.abs(kpoints_tab2))
        nk=[nk1,np.round(1/mk2),nk3]
    elif A==False and B==False and C==True:
        mk3=np.min(np.abs(kpoints_tab3))
        nk=[nk1,nk2,np.round(1/mk3)]
    else:
        nk=[nk1,nk2,nk3]
    
    return nk


#returns the dimension of the q+G sphere 
def GsphereSize(Grid):
    n1=np.amax(np.abs(Grid[:,0]))
    n2=np.amax(np.abs(Grid[:,1]))
    n3=np.amax(np.abs(Grid[:,2]))
    s1=int(2*n1+1)
    s2=int(2*n2+1)
    s3=int(2*n3+1)
    return s1,s2,s3


#Places all the value of the .SUS file for a given omega at the right index (leave some zeros the will be fill with symetrization)
def chi0GG_const(sus_ncfile,kpoints,nkpt,ng,nvec,omega):
    chi0GG=np.zeros((nvec,nvec),dtype=complex)
    print(nvec)
    for k in range(nkpt):
        chi0=sus_ncfile.reader.read_wggmat(kpoints[k]).wggmat
        #print('chi0[',k,']=',chi0[0])
        for i in range(ng):
            for j in range(ng):
                chi0GG[i*nkpt+k,j*nkpt+k]=chi0[omega,i,j]
    return chi0GG


#Returns the chi0 matrix with data symmetrized (chi0(G,G')=1/2(chi0(G,G')+chi0*(G',G)))
def chi0GGsym_const(vectoind,indtovec,nkpt,nqg,ng,nvec,chi0GG):
    chi0GGsym=np.zeros((nvec,nvec),dtype=complex)
    for p in range(nqg):
        for q in range(ng):
            i,j=q*nkpt+(p%nkpt),p
            qG=indtovec[i]
            qGprim=indtovec[j]
            m=vectoind[(-qG[0],-qG[1],-qG[2])]
            n=vectoind[(-qGprim[0],-qGprim[1],-qGprim[2])]
            chi0GGsym[i,j]=1/2*(chi0GG[i,j]+np.conj(chi0GG[m,n]))
            chi0GGsym[m,n]=1/2*(chi0GG[m,n]+np.conj(chi0GG[i,j]))
    return chi0GGsym


#Performs the FFT on the first coordinate on the whole matrix
def Gtor(nvec,fftboxsize,indtovec,n1,n2,n3,chi0GGsym):
    chi0rG=np.zeros((fftboxsize,nvec),dtype=complex)
    for i in range(nvec):
        #if i%200==0:
            #print(i/nvec*100,'percent of the first FFT is done')
        chi0rG[:,i]=FFT1(chi0GGsym[:,i],indtovec,n1,n2,n3,nvec,fftboxsize)
    return chi0rG



#Performs the FFT on the first coordinates. Places all the value corresponding to a given q+G2 in an FFT box and transfroms
def FFT1(column,indtovec,n1,n2,n3,nvec,fftboxsize):
    FFTSphere=np.zeros((n1,n2,n3),dtype=complex)
    FFT=np.zeros((n1,n2,n3),dtype=complex)
    columnout=np.zeros((fftboxsize),dtype=complex)
    for i in range(nvec):
        vec=indtovec[i]
        FFTSphere[vec[0],vec[1],vec[2]]=column[i]
    FFT=np.fft.ifftn(FFTSphere)
    columnout=np.reshape(FFT,fftboxsize)
    #print(columnout)
    return columnout


#Performs the FFT on the second coordinate on the whole matrix
def Gprimtorprim(nvec,fftboxsize,indtovec,n1,n2,n3,chi0rG):
    chi0rr=np.zeros((fftboxsize,fftboxsize),dtype=complex)
    for j in range(fftboxsize):
        #if j%200==0:
            #print(j/fftboxsize*100,'percent of the second FFT is done')
        chi0rr[j,:]=FFT2(chi0rG[j,:],indtovec,n1,n2,n3,nvec,fftboxsize)
    return chi0rr


#Performs the FFT on the first coordinates. Places all the value corresponding to a given r1 in an FFT box and transfroms
def FFT2(row,indtovec,n1,n2,n3,nvec,fftboxsize):
    FFTSphere=np.zeros((n1,n2,n3),dtype=complex)
    FFT=np.zeros((n1,n2,n3),dtype=complex)
    rowout=np.zeros(fftboxsize,dtype=complex)
    for i in range(nvec):
        vec=indtovec[i]
        FFTSphere[-vec[0],-vec[1],-vec[2]]=row[i]
    FFT=np.fft.ifftn(FFTSphere)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                rowout[k+j*n3+i*n3*n2]=FFT[i,j,k]
    return rowout

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


#Launch the different step for a given file with all the qpoint of the BZ and omega 
def Four6dp_opt(filename,omega=0):    
    sus_ncfile,kpoints,nw,ng,nkpt,G,nqg=openfile(filename)
    #print('Data extracted : ',ng,' G-vectors and ',nkpt,' q points (',nqg,' vectors to sample)')
    
    Grid=qG_Grid(nqg,kpoints,nkpt,ng,G)
    n1,n2,n3=GsphereSize(Grid)
    fftboxsize=n1*n2*n3
    #print('Gsphere contains ',len(Grid),'vectors')
    #print('The FFT box should at least have the dimensions (',n1,n2,n3,')')
    vectoind,indtovec=Dict_init(Grid,nqg)
    nvec=len(vectoind)
    #print('The dictionnaries are initialized')
    #Construction of chi0(q+G,q+G')
    chi0GG=chi0GG_const(sus_ncfile,kpoints,nkpt,ng,nvec,omega)
    #chi0GG,vectoind,indtovec,Grid=Build_chi0GG(kpoints,nsym,SymRec,TRec,G,ng)
    #Symmetrize chi0(q+G,q+G')
    chi0GGsym=chi0GGsym_const(vectoind,indtovec,nkpt,nqg,ng,nvec,chi0GG)
    #print(chi0GGsym)
    ##Takes each column of chi0(q+G,q+G'), transforms it and store it in chi0(r,q+G')
    chi0rG=Gtor(nvec,fftboxsize,indtovec,n1,n2,n3,chi0GGsym)
    ##Takes each row of chi0(r,q+G'), transforms it and store it in chi0(r,r')
    chi0rr=Gprimtorprim(nvec,fftboxsize,indtovec,n1,n2,n3,chi0rG)
    chi0rr_out=np.reshape(chi0rr,(n1,n2,n3,n1,n2,n3))
    return chi0rr_out

def Four6dp_opt_sym(filename,omega=0):    
    sus_ncfile,kpoints,nw,ng,nkpt,G,nqg=openfile(filename)
    kpoints=np.array([[0,0,0],[0.25,0,0],[0.5,0,0],[0.25,0.25,0],[0.5,0.25,0],[-0.25,0.25,0],[0.5,0.5,0],[-0.25,0.5,0.25]])
    #print('Data extracted : ',ng,' G-vectors and ',nkpt,' q points (',nqg,' vectors to sample)')
    SymRec,TRec,Tim_Rev,nsym=Get_Sym(filename)
    chi0gg,Gvec,vectoind_q,indtovec_q=Chi0GG_sym(sus_ncfile,kpoints,nsym,SymRec,TRec,G,ng,Tim_Rev)
    print(chi0gg.shape)
    ngvec=len(Gvec)
    nq=len(vectoind_q)
    nqgvec=nq*ngvec
    print(nqgvec)
    Grid=np.zeros((nqgvec,3),dtype=int)
    n=4
    for i in range(nq):
        qvec=indtovec_q[i]
        for j in range(ngvec):
            Grid[j+i*ngvec]=[4*(qvec[0]+Gvec[j,0]),4*(qvec[1]+Gvec[j,1]),4*(qvec[2]+Gvec[j,2])]
    #print(Grid)
    n1,n2,n3=GsphereSize(Grid)
    fftboxsize=n1*n2*n3
    print(Grid.shape)
    #print('Gsphere contains ',len(Grid),'vectors')
    #print('The FFT box should at least have the dimensions (',n1,n2,n3,')')
    vectoind,indtovec=Dict_init(Grid,nqgvec)
    #print(vectoind)
    nvec=len(vectoind)
    #print(nvec)
    #print('The dictionnaries are initialized')
    #Construction of chi0(q+G,q+G')
    chi0GG2D=np.zeros((nvec,nvec),dtype=complex)
    for i in range(nq):
        for j in range(ngvec):
            for k in range(ngvec):
                chi0GG2D[j*nq+i,k*nq+i]=chi0gg[i,j,k]
    #chi0GG=chi0GG_const(sus_ncfile,kpoints,nkpt,ng,nvec,omega)
    #chi0GG,vectoind,indtovec,Grid=Build_chi0GG(kpoints,nsym,SymRec,TRec,G,ng)
    #Symmetrize chi0(q+G,q+G')
    chi0GGsym=chi0GGsym_const(vectoind,indtovec,nq,nqgvec,ngvec,nvec,chi0GG2D)
    print(chi0GGsym)
    ##Takes each column of chi0(q+G,q+G'), transforms it and store it in chi0(r,q+G')
    chi0rG=Gtor(nvec,fftboxsize,indtovec,n1,n2,n3,chi0GGsym)
    ##Takes each row of chi0(r,q+G'), transforms it and store it in chi0(r,r')
    chi0rr=Gprimtorprim(nvec,fftboxsize,indtovec,n1,n2,n3,chi0rG)
    chi0rr_out=np.reshape(chi0rr,(n1,n2,n3,n1,n2,n3))
    return chi0rr_out

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################

def Build_GDict(G,ng):
    G_dict={}
    for i in range(ng):
        G_dict[(G[i,0],G[i,1],G[i,2])]=i
    return G_dict

def Build_QpointList(indtovec_q,nqpt):
    qpoints=np.zeros((nqpt,3))
    for i in range(nqpt):
        qi=indtovec_q[i]
        qpoints[i]=[qi[0],qi[1],qi[2]]
    return qpoints

def Build_BZ(kpt,nsym,SymRec,TRec):
    vectoind_q,indtovec_q,sym_dict={},{},{}
    nkpt_test=len(kpt)
    ind=0
    for i in range(nsym):
        for j in range(nkpt_test):
            q=np.matmul(SymRec[i],kpt[j]+TRec[i])
            if (q[0],q[1],q[2]) not in vectoind_q.keys():
                if np.amax(q)<=0.5 and np.amin(q)>-0.5:
                    vectoind_q[(q[0],q[1],q[2])]=ind
                    indtovec_q[ind]=(q[0],q[1],q[2])
                    sym_dict[ind]=(i,j,0,0,0)
                    ind+=1
                else:
                    G0,G1,G2=0,0,0
                    if q[0]>0.5:
                        G0=-1
                    elif q[0]<=-0.5:
                        G0=1
                    if q[1]>0.5:
                        G1=-1
                    elif q[1]<=-0.5:
                        G1=1
                    if q[2]>0.5:
                        G2=-1
                    elif q[2]<=-0.5:
                        G2=1
                    Gi=[G0,G1,G2]
                    q=q+Gi
                    if (q[0],q[1],q[2]) not in vectoind_q.keys():
                    #print('q was added to the list after addition of an umklapp vector')
                        vectoind_q[(q[0],q[1],q[2])]=ind
                        indtovec_q[ind]=(q[0],q[1],q[2])
                        sym_dict[ind]=(i,j,G0,G1,G2)
                        ind+=1           
    return sym_dict,vectoind_q,indtovec_q

def Build_Gsphere(G,ng):
    G_dict={}
    ind_dict={}
    for i in range(ng):
        G_dict[(G[i,0],G[i,1],G[i,2])]=i
        ind_dict[i]=[G[i,0],G[i,1],G[i,2]]
    return G_dict,ind_dict

def Build_chi0GG(kpt,nsym,SymRec,TRec,G,ng):
    #STEP1 : Recreate the BZ
    sym_dict,vectoind_q,indtovec_q=Build_BZ(kpt,nsym,SymRec,TRec)
    nqpt=len(sym_dict)
    #STEP2 : Create dictionnaries with the G vectors and q+G vectors
    
    G_dict=Build_GDict(G,ng)
    #Gcomp,indcomp=Build_Gsphere(G,ng)
    #ngcomp=len(Gcomp)
    qpoints=Build_QpointList(indtovec_q,nqpt)
    nqg=nqpt*ng
    Grid_ind=qG_Grid_sym(nqg,qpoints,nqpt,ng,G)
    #STEP3 : Create dictionnaries to keep track of the q+G vectord and their place in the table
    vectoind,indtovec=Dict_init(Grid_ind,nqg)
    #print(vectoind)
    #STEP4 : Build chi0 using equation in above cell
    nvec=len(vectoind)
    chi0GG=np.zeros((nvec,nvec),dtype=complex)
    ic=complex(0,1)
    mk1,mk2,mk3=FindSmallerKpoint(qpoints)
    nk=[round(1/mk1),round(1/mk2),round(1/mk3)]
    for i in range(nqpt):
        sym_data=sym_dict[i]
        #print(qpoints[sym_data[1]])
        chi0=sus_ncfile.reader.read_wggmat(qpoints[sym_data[1]]).wggmat[0]
        sym=sym_data[0]
        t=TRec[sym]
        S=np.linalg.inv(SymRec[sym])
        qvec=qpoints[i]
        #print(qvec)
        for j in range(ng):
            G1=G[j]
            qG1=np.multiply((qvec+G1),nk)
            ind1=vectoind[(qG1[0],qG1[1],qG1[2])]
            G1Gs=G1#+[sym_data[2],sym_data[3],sym_data[4]]
            Gind1=np.matmul(S,G1Gs)
            indchi1=G_dict[(Gind1[0],Gind1[1],Gind1[2])]
            for k in range(ng):
                G2=G[k]
                qG2=np.multiply((qvec+G2),nk)
                ind2=vectoind[(qG2[0],qG2[1],qG2[2])]
                G2Gs=G2#+[sym_data[2],sym_data[3],sym_data[4]]
                Gind2=np.matmul(S,G2Gs)
                indchi2=G_dict[(Gind2[0],Gind2[1],Gind2[2])]
                chi0GG[ind1,ind2]=cmath.exp(ic*np.dot(t,G2-G1))*chi0[indchi1,indchi2]
    return chi0GG,vectoind,indtovec,Grid_ind


def Dict_initKaltak(Grid,nqg):
    vectoind={}
    indtovec={}
    newind=nqg
    for i in range(nqg):
        vectoind[(Grid[i,0],Grid[i,1],Grid[i,2])]=i
        indtovec[i]=(Grid[i,0],Grid[i,1],Grid[i,2]) 
    return vectoind,indtovec

def Gsym(G,ng):
    Gdict={}
    IndG={}
    for i in range(ng):
        Gdict[(G[i,0],G[i,1],G[i,2])]=i
        IndG[i]=(G[i,0],G[i,1],G[i,2])
    return Gdict,IndG

def SymGG(sus_ncfile,kpoints,nkpt,ng,omega,Gdict,IndDict):
    chi0=np.zeros((nkpt,ng,ng),dtype=complex)
    for i in range(nkpt):
        chi0q=sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat[omega]
        chi0[i,:,:]=chi0q  
    return chi0


def Chi0GG_sym(sus_ncfile,kpt,nsym,SymRec,TRec,G,ng,Tim_Rev,omega=0):
    #Step1: Construction of BZ
    sym_dict,vectoind_q,indtovec_q=Build_BZ(kpt,nsym,SymRec,TRec)
    nqpt=len(sym_dict)
    nkpt=len(kpt)
    #Step2: Gdict
    G_dict_start=Build_GDict(G,ng)
    #print(G_dict_start.keys())
    #Step3: Gdict useful
    G_dict_sym={}
    ind=0
    for j in range(ng):
        G_dict_sym[j]=1
    for i in range(nqpt):
        sym_data=sym_dict[i]
        rot=SymRec[sym_data[0]]
        umklp=[sym_data[2],sym_data[3],sym_data[4]]
        for j in range(ng):
            vec=(G[j,0],G[j,1],G[j,2])
            G_f=np.matmul(np.linalg.inv(rot),G[j]+umklp)
            vectest=(G_f[0],G_f[1],G_f[2])
            if vectest not in G_dict_start.keys() and G_dict_sym[j]!=-1:
                #print(str(vec),'with symmetry',str(rot), 'gives the vector outside the sphere',str(vectest))
                G_dict_sym[j]=-1
    

    count=0
    Gvec_ind=[]
    G_dict_vectoind={}
    for i in range(ng):
        if G_dict_sym[i]==1:
            G_dict_vectoind[(G[i,0],G[i,1],G[i,2])]=count
            count+=1
            Gvec_ind=np.append(Gvec_ind,[i])        
    Gvec=np.zeros((count,3),dtype=int)
    chi0gg=np.zeros((nqpt,count,count),dtype=complex)
    for i in range(nkpt):
        ind1=-1
        chi0=sus_ncfile.reader.read_wggmat(kpt[i]).wggmat[omega]
        for j in Gvec_ind:
            ind1+=1
            ind3=int(j)
            ind2=0
            Gvec[ind1]=G[ind3]
            for k in Gvec_ind:
                ind4=int(k)
                chi0gg[i,ind1,ind2]=chi0[ind3,ind4]
                ind2+=1
    ic=complex(0,1)
    for i in range(nkpt,nqpt):
        sym_data=sym_dict[i]
        rot=SymRec[sym_data[0]]
        t=TRec[sym_data[0]]
        invrot=np.linalg.inv(rot)
        umklp=[sym_data[2],sym_data[3],sym_data[4]]
        chi0=sus_ncfile.reader.read_wggmat(kpt[sym_data[2]]).wggmat[omega]
        for j in range(count):
            Grot1=np.matmul(invrot,(Gvec[j]+umklp))
            indgrot1=G_dict_start[(Grot1[0],Grot1[1],Grot1[2])]
            for k in range(count):
                Grot2=np.matmul(invrot,(Gvec[k]+umklp))
                indgrot2=G_dict_start[(Grot2[0],Grot2[1],Grot2[2])]
                chi0gg[i,j,k]=cmath.exp(ic*np.dot(t,Gvec[k]-Gvec[j]))*chi0[indgrot1,indgrot2]
    if Tim_Rev==True:
        chi0ggTR=chi0gg
        IndtoAvoid={}
        count1=count
        for i in range(count):
            if (-Gvec[i,0],-Gvec[i,1],-Gvec[i,2]) not in G_dict_vectoind.keys():
                G_dict_vectoind[(-Gvec[i,0],-Gvec[i,1],-Gvec[i,2])]=count1
                Gvec=np.append(Gvec,[[-Gvec[i,0],-Gvec[i,1],-Gvec[i,2]]],axis=0)
                IndtoAvoid[i]=1
                count1+=1
        chi0gg=np.zeros((nqpt,count1,count1),dtype=complex)
        
        for i in range(nqpt):
            chi0gg[i,0:count,0:count]=chi0ggTR[i]    
            qvec=indtovec_q[i]
            qrev=[-qvec[0],-qvec[1],-qvec[2]]
            if qrev[0]==-0.5:
                qrev[0]=0.5
            if qrev[1]==-0.5:
                qrev[1]=0.5
            if qrev[2]==-0.5:
                qrev[2]=0.5
            ind1=vectoind_q[(qrev[0],qrev[1],qrev[2])]
            for j in range(count1):
                if j in IndtoAvoid.keys():
                    continue    
                else:
                    ind3=G_dict_vectoind[(-Gvec[j,0],-Gvec[j,1],-Gvec[j,2])]
                for k in range(count1):
                    if j<count & k<count:
                        continue
                    elif k in IndtoAvoid.keys():
                        continue
                    else:
                        ind2=G_dict_vectoind[(-Gvec[k,0],-Gvec[k,1],-Gvec[k,2])]
                        chi0gg[i,j,k]=chi0ggTR[ind1,ind2,ind3]
    
    return chi0gg,Gvec,vectoind_q,indtovec_q

########################################################################################
########################################################################################
########################################################################################
########################################################################################
########################################################################################


def FFTKaltak1(chi0symGG,n1,n2,n3,nkpt,ng,nfft1,Gdict,IndDict):
    chi0rG=np.zeros((nkpt,nfft1,ng),dtype=complex)
    for i in range(nkpt):   
        for j in range(ng):
            FFTbox=np.zeros((n1,n2,n3),dtype=complex)
            FFT=np.zeros((n1,n2,n3),dtype=complex)
            for k in range(ng):
                Gvec=IndDict[k]
                #print(Gvec)
                FFTbox[Gvec[0],Gvec[1],Gvec[2]]=chi0symGG[i,j,k]
            FFT=np.fft.ifftn(FFTbox)
            chi0rG[i,:,j]=np.reshape(FFT,nfft1)
            for l in range(n1):
                for m in range(n2):
                    for n in range(n3):
                        chi0rG[i,n+m*n3+l*n3*n2,j]=FFT[l,m,n]
    return chi0rG


def FFTKaltak2(chi0symrG,n4,n5,n6,nkpt,ng,nfft1,nfft2,Gdict,IndDict):
    chi0rR=np.zeros((nfft1,nfft2),dtype=complex)
    for i in range(nfft1):
        FFTbox=np.zeros((n4,n5,n6),dtype=complex)
        FFT=np.zeros((n4,n5,n6),dtype=complex)
        for j in range(nkpt):
            for k in range(ng):
                Gvec=IndDict[k+j*ng]
                FFTbox[-Gvec[0],-Gvec[1],-Gvec[2]]=chi0symrG[j,i,k]
        FFT=np.fft.ifftn(FFTbox)  
        for l in range(n4):
            for m in range(n5):
                for n in range(n6):
                    chi0rR[i,(n+m*n6+l*n6*n5)]=FFT[-l,-m,-n]
    return chi0rR


def KaltakAlgorithm(filename,omega=0):
    sus_ncfile,kpoints,nw,ng,nkpt,G,nqg=openfile(filename)
    #returns the dictionnary of G-vectors
    Gdict,IndDict=Gsym(G,ng)
    
    #Stock toutes les données dans Matrice 64x113x113
    chi0symGG=np.zeros((nkpt,ng,ng),dtype=complex)
    #chi0symGG=SymGG(sus_ncfile,kpoints,nkpt,ng,omega,Gdict,IndDict)
    for i in range(nkpt):
        chi0symGG[i]=sus_ncfile.reader.read_wggmat(kpoints[i]).wggmat[omega]
    #Détermine les dimensions de la Box FFT
    n1,n2,n3=np.amax(np.abs(G[:,0]))*2+1,np.amax(np.abs(G[:,1]))*2+1,np.amax(np.abs(G[:,2]))*2+1
    nfft1=n1*n2*n3
    #Réalise le FFT sur le colonnes de chaque sous matrices (output : 64,343,113)
    chi0rG=FFTKaltak1(chi0symGG,n1,n2,n3,nkpt,ng,nfft1,Gdict,IndDict)
    #initialise le grille de points q+G
    nqg=ng*nkpt
    Grid=qG_Grid(nqg,kpoints,nkpt,ng,G)
    #Détermine les dimensions de la deuxièmes box FFT
    n4,n5,n6=GsphereSize(Grid)
    n4,n5,n6=n4-1,n5-1,n6-1
    print(n4,n5,n6)
    nfft2=n4*n5*n6
    
    vectoind,indtovec=Dict_initKaltak(Grid,nqg)
    #Réalise la seconde FFT en récupérant tous les points d'un même rp (output : 343x21952)
    chi0rR=FFTKaltak2(chi0rG,n4,n5,n6,nkpt,ng,nfft1,nfft2,vectoind,indtovec)
    chi0=np.reshape(chi0rR,(n1,n2,n3,n4,n5,n6))
    return chi0

def Get_Sym(filename):
    structure=abipy.core.structure.Structure.from_file(filename)
    #print(structure)
    Sym=abipy.core.symmetries.AbinitSpaceGroup.from_structure(structure)
    SymRec=Sym.symrec
    TRec=Sym.tnons
    Tim_Rev=Sym.has_timerev
    nsym=len(SymRec)
    return SymRec,TRec,Tim_Rev,nsym


def KaltakAlgorithm_fromSym(filename,omega=0):
    sus_ncfile,kpoints,nw,ng,nkpt,G,nqg=openfile(filename)
    kpoints=np.array([[0,0,0],[0.25,0,0],[0.5,0,0],[0.25,0.25,0],[0.5,0.25,0],[-0.25,0.25,0],[0.5,0.5,0],[-0.25,0.5,0.25]])
    
    
    SymRec,TRec,Tim_Rev,nsym=Get_Sym(filename)
    chi0gg,Gvec,vectoind_q,indtovec_q=Chi0GG_sym(sus_ncfile,kpoints,nsym,SymRec,TRec,G,ng,Tim_Rev,omega=0)
    maxG1,maxG2,maxG3=np.amax(np.abs(Gvec[:,0])),np.amax(np.abs(Gvec[:,1])),np.amax(np.abs(Gvec[:,2]))
    n1,n2,n3=int(maxG1*2+1),int(maxG2*2+1),int(maxG3*2+1)
    nfft1=int(n1*n2*n3)
    ngvec=len(Gvec)
    s1,s2,s3=chi0gg.shape
    #FFT1
    chi0rg=np.zeros((s1,nfft1,s3),dtype=complex)
    for i in range(s1):
        FFTbox1=np.zeros((n1,n2,n3),dtype=complex)
        FFTout1=np.zeros((n1,n2,n3),dtype=complex)
        for j in range(s3):
            for k in range(s2):
                FFTbox1[Gvec[k,0],Gvec[k,0],Gvec[k,0]]=chi0gg[i,k,j]
            FFTout1=np.fft.ifftn(FFTbox1)
            chi0rg[i,:,j]=np.reshape(FFTout1,nfft1)
    #print(chi0rg)
    nqg=len(vectoind_q)*ngvec
    #kpoint_tab=kpoints.frac_coords
    ind=kpoints!=0
    kpoints=kpoints[ind]
    qstepmin=np.amin(np.abs(kpoints))
    qnormmax=np.amax(np.abs(kpoints))
    n=1/qstepmin
    n4,n5,n6=2*int(n*(maxG1+qnormmax)),2*int(n*(maxG2+qnormmax)),2*int(n*(maxG3+qnormmax))
    nfft2=int(n4*n5*n6)
    chi0rr=np.zeros((nfft1,nfft2),dtype=complex)
    for i in range(nfft1):
        FFTbox2=np.zeros((n4,n5,n6))
        for j in range(s1):
            qvec=indtovec_q[j]
            for k in range(s3):
                qGvec=n*(qvec+Gvec[k])
                FFTbox2[-int(qGvec[0]),-int(qGvec[1]),-int(qGvec[2])]=chi0rg[j,i,k]
            FFTout2=np.fft.ifftn(FFTbox2)
            chi0rr[i,:]=np.reshape(FFTout2,nfft2)
    
    chi0rr=np.reshape(chi0rr,(n1,n2,n3,n4,n5,n6))
    return chi0rr

def chi0rrsymKaltak(chi0rr):
    s1,s2,s3,s4,s5,s6=chi0rr.shape
    n1,n2,n3,n4,n5,n6=s1,s2,s3,s4,s5,s6
    chi0rrsym=np.zeros((n1,n2,n3,n4,n5,n6),dtype=complex)
    for i in range(n1):
        for j in range(n2):
            for k in range(n3):
                for l in range(n4):
                    for m in range(n5):
                        for n in range(n6):
                            rs=[l,m,n]
                            rp=[l%n1,m%n2,n%n3]
                            R=[rs[0]-rp[0],rs[1]-rp[1],rs[2]-rp[2]]
                            chi0rrsym[i,j,k,l,m,n]=1/2*(chi0rr[i,j,k,l,m,n]+chi0rr[rp[0],rp[1],rp[2],i-R[0],j-R[1],k-R[2]])
    return chi0rrsym