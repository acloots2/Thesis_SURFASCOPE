from Fourier_tool import openfile
from Fourier_tool import FindSmallerKpoint as fsk
from abipy.electrons.scr import ScrFile
import numpy as np
import math
import pointcloud as pc
import plotly.graph_objects as go
from scipy.interpolate import RegularGridInterpolator
import DRF
import Second_P_models as SP


class Dielectric_function_slab:
    def __init__(self, filename, qp, point_dens, omega, d, eta, n, sym = False):
        self.q_paral = qp
        self.thickness = d
        self.spat_coord1 = np.linspace(0, round(d/2), point_dens*round(d/2)+1)
        self.spat_coord2 = np.linspace(0, round(d), point_dens*round(d)+1)
        self.freq = omega
        self.nfreq = len(omega)
        self.nz1 = len(self.spat_coord1)
        self.nz2 = len(self.spat_coord2)
        self.qvec = SP.zvec_to_qvec(self.spat_coord2)
        self.elec_density = n
        self.damping = eta*0.03675
        self.plasma_freq = math.sqrt(self.elec_density*4*math.pi)*27.211
        self.dens_resp_func = []
        self.sym = sym
        self.permittivity = []
        self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps, self.loss_func = [], [], [], []
        self.dens_resp_func_recip = []
        self.weight = []
        if filename == []:
            self.dens_resp_func = DRF.chi0wzz_slab_jellium_Eguiluz_1step_F(self.q_paral, self.spat_coord1, self.spat_coord2, self.freq, self.elec_density, self.thickness, self.damping)
            filename = "chi0wzz_slab_"+str(qp*1000)+"_"+str(point_dens)+"_"+str(min(omega)*100)+"_"+str(max(omega)*100)+"_"+str(self.nfreq)+"_"+str(d)+"_"+str(eta)+"_"+str(n*10000)
            np.save(filename, self.dens_resp_func)
        else:
            self.dens_resp_func = np.load(filename)
            nw_test, nz1_test, nz2_test = self.dens_resp_func.shape
            if self.sym == False:
                if self.nfreq != nw_test or self.nz1!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")
            else:
                if self.nfreq != nw_test or self.nz2!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")

    def sym_drf(self):
        if self.sym == True:
            raise ValueError("The function is already symmetrized")
        self.sym = True
        self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)

    def get_freq(self):
        return self.freq
    
    def get_qp(self):
        return self.q_paral

    def get_thickness(self):
        return self.thickness
    
    def get_spat_coord(self, x):
        if x==0:
            return self.spat_coord1
        return self.spat_coord2
    
    def get_nfreq(self):
        return self.nfreq
    
    def get_nz(self, x):
        if x==0:
            return self.nz1
        return self.nz2

    def get_elec_dens(self):
        return self.elec_density

    def get_damping(self):
        return self.damping

    def get_dens_resp_func(self):
        return self.dens_resp_func

    def get_permittivity(self):
        if self.permittivity==[]:
            if self.sym == False:
                self.sym = True
                self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)
                return SP.epsilon_Wilson(self.dens_resp_func, self.qvec, self.q_paral, opt = "Slab")
            else:
                return SP.epsilon_Wilson(self.dens_resp_func, self.qvec, self.q_paral, opt = "Slab")
        else:
            return self.permittivity
    
    def add_recip_dens_resp_func(self):
        if self.sym == False:
            self.sym = True
            self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)
            self.dens_resp_func_recip = SP.Fourier_inv(self.dens_resp_func, self.spat_coord2)
        self.dens_resp_func_recip = SP.Fourier_inv(self.dens_resp_func, self.spat_coord2)

    def add_permittivity(self):
        if self.dens_resp_func_recip == []:
            self.add_recip_dens_resp_func()
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")
        else:
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")

    def add_permittivity_hermi(self):
        if self.dens_resp_func_recip == []:
            self.add_recip_dens_resp_func()
            chi_out = np.zeros((self.nfreq, self.nz2, self.nz2), dtype = complex)
            for i in range(self.nfreq):
                chi_out[i] = (self.dens_resp_func_recip[i]+np.conj(self.dens_resp_func_recip[i]))/2
            self.dens_resp_func_recip = chi_out
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")
        else:
            chi_out = np.zeros((self.nfreq, self.nz2, self.nz2), dtype = complex)
            for i in range(self.nfreq):
                chi_out[i] = (self.dens_resp_func_recip[i]+np.conj(self.dens_resp_func_recip[i]))/2
            self.dens_resp_func_recip = chi_out
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")
         
    
    def add_eigen_values(self):
        if self.permittivity == []:
            self.add_permittivity()
            self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps = SP.eig_plasmons(self.permittivity)
        else:
            self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps = SP.eig_plasmons(self.permittivity)
    
    def add_weight(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            self.weight = SP.weights(self.eig_rvec_eps, self.eig_lvec_eps)
        else:
            self.weight = SP.weights(self.eig_rvec_eps, self.eig_lvec_eps)
    
    def add_loss_func(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            self.loss_func = SP.Loss_Func_final(self.eig_v_eps, self.eig_rvec_eps, self.eig_lvec_eps)
        else:
            self.loss_func = SP.Loss_Func_final(self.eig_v_eps, self.eig_rvec_eps, self.eig_lvec_eps)

   

    def add_loss_func_test1(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            return SP.Loss_Func(self.eig_v_eps)
        return SP.Loss_Func(self.eig_v_eps)


    def add_loss_func_Majerus(self):
        if self.permittivity == []:
            self.add_permittivity()
        return SP.Loss_Func_Majerus(self.permittivity)

class Dielectric_function_bulk:
    def __init__(self, filename, point_dens, omega, d, eta, n, sym = False):
        self.thickness = d
        self.spat_coord1 = np.linspace(0, round(d/2), point_dens*round(d/2)+1)
        self.spat_coord2 = np.linspace(0, round(d), point_dens*round(d)+1)
        self.freq = omega
        self.nfreq = len(omega)
        self.nz1 = len(self.spat_coord1)
        self.nz2 = len(self.spat_coord2)
        self.qvec = SP.zvec_to_qvec(self.spat_coord2)
        self.elec_density = n
        self.damping = eta*0.03675
        self.dens_resp_func = []
        self.sym = sym
        self.q_paral = 0
        self.permittivity = []
        self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps, self.loss_func = [], [], [], []
        self.dens_resp_func_recip = []
        self.weight = []
        self.plasma_freq = math.sqrt(self.elec_density*4*math.pi)*27.211
        if filename == []:
            self.dens_resp_func_recip = SP.chi0q_XG(self.qvec, self.freq, self.elec_density)
            filename = "chi0wG_bulk_"+str(point_dens)+"_"+str(min(omega)*100)+"_"+str(max(omega)*100)+"_"+str(self.nfreq)+"_"+str(d)+"_"+str(eta)+"_"+str(n*10000)
            np.save(filename, self.dens_resp_func_recip)
        else:
            self.dens_resp_func_recip = np.load(filename)
            nw_test, nz1_test, nz2_test = self.dens_resp_func_recip.shape
            if self.sym == False:
                if self.nfreq != nw_test or self.nz1!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")
            else:
                if self.nfreq != nw_test or self.nz2!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")
        
    def sym_drf(self):
        if self.sym == True:
            raise ValueError("The function is already symmetrized")
        self.sym = True
        self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)

    def get_freq(self):
        return self.freq

    def get_thickness(self):
        return self.thickness
    
    def get_spat_coord(self, x):
        if x==0:
            return self.spat_coord1
        return self.spat_coord2
    
    def get_nfreq(self):
        return self.nfreq
    
    def get_nz(self, x):
        if x==0:
            return self.nz1
        return self.nz2

    def get_elec_dens(self):
        return self.elec_density

    def get_damping(self):
        return self.damping

    def get_dens_resp_func(self):
        return self.dens_resp_func

    def get_permittivity(self):
        if self.permittivity==[]:
            if self.sym == False:
                self.sym = True
                self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)
                return SP.epsilon_Wilson(self.dens_resp_func, self.qvec, self.q_paral, opt = "Bulk")
            else:
                return SP.epsilon_Wilson(self.dens_resp_func, self.qvec, self.q_paral, opt = "Bulk")
        else:
            return self.permittivity
    
    def add_dens_resp_func(self):
        if self.sym == False:
            self.sym = True
            self.dens_resp_func = SP.Sym_chi_Slab(self.dens_resp_func)
            self.dens_resp_func_recip = SP.chi0zz_XG(self.spat_coord2, self.freq, self.elec_density)
        self.dens_resp_func_recip = SP.chi0zz_XG(self.spat_coord2, self.freq, self.elec_density)

    def add_permittivity(self):
        if self.dens_resp_func_recip == []:
            self.add_recip_dens_resp_func()
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, self.qvec, self.q_paral, opt = "Bulk")
        else:
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, self.qvec, self.q_paral, opt = "Bulk")

    def add_permittivity_hermi(self):
        if self.dens_resp_func_recip == []:
            self.add_recip_dens_resp_func()
            chi_out = np.zeros((self.nfreq, self.nz2, self.nz2), dtype = complex)
            for i in range(self.nfreq):
                chi_out[i] = (self.dens_resp_func_recip[i]+np.conj(self.dens_resp_func_recip[i]))/2
            self.dens_resp_func_recip = chi_out
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")
        else:
            chi_out = np.zeros((self.nfreq, self.nz2, self.nz2), dtype = complex)
            for i in range(self.nfreq):
                chi_out[i] = (self.dens_resp_func_recip[i]+np.conj(self.dens_resp_func_recip[i]))/2
            self.dens_resp_func_recip = chi_out
            self.permittivity = SP.epsilon_Wilson(self.dens_resp_func_recip, np.real(SP.Inv_Rev_vec(self.qvec)), self.q_paral, opt = "Slab")
         
    
    def add_eigen_values(self):
        if self.permittivity == []:
            self.add_permittivity()
            self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps = SP.eig_plasmons(self.permittivity)
        else:
            self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps = SP.eig_plasmons(self.permittivity)
    
    def add_weight(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            self.weight = SP.weights(self.eig_rvec_eps, self.eig_lvec_eps)
        else:
            self.weight = SP.weights(self.eig_rvec_eps, self.eig_lvec_eps)
    
    def add_loss_func(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            self.loss_func = SP.Loss_Func_final(self.eig_v_eps, self.eig_rvec_eps, self.eig_lvec_eps)
        else:
            self.loss_func = SP.Loss_Func_final(self.eig_v_eps, self.eig_rvec_eps, self.eig_lvec_eps)


    def add_loss_func_test1(self):
        if self.eig_v_eps == []:
            self.add_eigen_values()
            return SP.Loss_Func(self.eig_v_eps)
        return SP.Loss_Func(self.eig_v_eps)

    





    
    

    


            


