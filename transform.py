import numpy as np
import math 
import cmath
import abipy
from abipy.electrons.scr import ScrFile
import time
import matplotlib.pyplot as plt
import Fourier_tool as Ft
import XGChi0 
import json

if __name__ == '__main__':
    filename = "Slab5Atoms6o_DS1_SUS.nc"
    print(filename)
    #BUILD
    chi0GG, ind_qG_to_vec, n1, n2, n3, ind_q_to_vec, ind_G_to_vec, G, nk = XGChi0.Build_Chi0GG(filename, "FromSym", omega = 0)
    #print(n1, n2, n3)   
    np.save("chi0GGslab5", chi0GG)
    np.save("dim", np.array([n1, n2, n3]))
    np.save("Garray", G)
    np.save("nk", nk)
    jsonqG = json.dumps(ind_qG_to_vec)
    jsonq = json.dumps(ind_q_to_vec)
    jsonG = json.dumps(ind_G_to_vec)
    fqG = open("qGvec.json", "w")
    fqG.write(jsonqG)
    fqG.close()
    fq = open("qvec.json", "w")
    fq.write(jsonq)
    fq.close()
    fG = open("Gvec.json", "w")
    fG.write(jsonG)
    fG.close()
