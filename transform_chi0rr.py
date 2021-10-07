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


if __name__ == "__main__":
    chi0GG = np.load("chi0GGslab5.npy")
    G = np.load("Garray.npy")
    nk = np.load("nk.npy")
    ind_qG_to_vec = json.load("qGvec.json")
    ind_G_to_vec = json.load("Gvec.json")
    ind_q_to_vec = json.load("qvec.json")
    chi0rr = XGChi0.FFT_chi0_sizeadapt(np.real(chi0GG), ind_qG_to_vec, 72, 72, 60, 12, 12, 60, ind_q_to_vec, ind_G_to_vec, G, nk, opt2 = "Kaltak")
    np.save("chi0rr_slab5atoms6_ecut8", chi0rr)