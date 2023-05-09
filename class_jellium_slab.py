"""Class that builds an object containing all the required data to analyze the 
dielectric properties of a given jellium slab"""

import math
import numpy as np
import jellium_slab as js
import tools
import test_set as ts


class DielFuncSlab:
    """Class with the slab object for which different dielectric properties can be computed"""
    def __init__(self, filename, q_p, point_dens, omega, d_slab, eta, dens, sym = False):
        self.q_paral = q_p
        self.thickness = d_slab
        self.spat_coord1 = np.linspace(0, round(d_slab/2), point_dens*round(d_slab/2)+1)
        self.spat_coord2 = np.linspace(0, round(d_slab), point_dens*round(d_slab)+1)
        self.freq = omega
        self.nfreq = len(omega)
        self.nz1 = len(self.spat_coord1)
        self.nz2 = len(self.spat_coord2)
        self.qvec = tools.zvec_to_qvec(self.spat_coord2)
        self.elec_density = dens
        self.damping = eta*0.03675
        self.plasma_freq = math.sqrt(self.elec_density*4*math.pi)*27.211
        self.dens_resp_func = []
        self.sym = sym
        self.dielectric_resp = []
        self.eig_v_eps, self.eig_lvec_eps, self.eig_rvec_eps, self.loss_func = [], [], [], []
        self.dens_resp_func_recip = []
        self.weight = []
        if filename == []:
            self.dens_resp_func = js.chi0wzz_slab_jellium(self.q_paral, self.spat_coord1, self.spat_coord2, self.freq, self.elec_density, self.thickness, self.damping)
            filename = "chi0wzz_slab_"+str(q_p*1000)+"_"+str(point_dens)+"_"+str(min(omega)*100)+"_"+str(max(omega)*100)+"_"+str(self.nfreq)+"_"+str(d_slab)+"_"+str(eta)+"_"+str(dens*10000)
            np.save(filename, self.dens_resp_func)
        else:
            self.dens_resp_func = np.load(filename)
            nw_test, nz1_test, nz2_test = self.dens_resp_func.shape
            if not self.sym:
                if self.nfreq != nw_test or self.nz1!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")
            else:
                if self.nfreq != nw_test or self.nz2!= nz1_test or self.nz2 != nz2_test:
                    raise ValueError("The file you are trying to load does not seem to correspond the specifications given, check the sym option first")

    def sym_drf(self):
        """Symmetrizes in real space the density response function"""
        if self.sym:
            raise ValueError("The function is already symmetrized")
        self.sym = True
        self.dens_resp_func = js.sym_chi_slab(self.dens_resp_func)

    def get_freq(self):
        """Return the frequency domain"""
        return self.freq
    
    def get_q_p(self):
        """Return the component parallel to the surface of the perturbation"""
        return self.q_paral

    def get_thickness(self):
        """Return the size of the slab"""
        return self.thickness
    
    def get_spat_coord(self, spc):
        """return the first or second spatial coordinate"""
        if spc==0:
            return self.spat_coord1
        return self.spat_coord2
    
    def get_nfreq(self):
        """return the number of frequencies"""
        return self.nfreq
    
    def get_nz(self, spc):
        """return the number of points used to sample the real space of the first or second spat coord"""
        if spc==0:
            return self.nz1
        return self.nz2

    def get_elec_dens(self):
        """return the electronic density used"""
        return self.elec_density

    def get_damping(self):
        """return the damping in Ha"""
        return self.damping

    def get_dens_resp_func(self):
        """return chi0qwzz"""
        return self.dens_resp_func

    def get_dielectric_resp(self):
        """Return the dielectric response"""
        if self.dielectric_resp==[]:
            if not self.sym:
                self.sym = True
                self.dens_resp_func = js.sym_chi_slab(self.dens_resp_func)
                return js.epsilon(self.dens_resp_func, self.qvec, self.q_paral)
            else:
                return js.epsilon(self.dens_resp_func, self.qvec, self.q_paral)
        else:
            return self.dielectric_resp
    
    def add_recip_dens_resp_func(self):
        """Computes chi0qwgg from chi0qwzz"""
        if not self.sym:
            self.sym = True
            self.dens_resp_func = js.sym_chi_slab(self.dens_resp_func)
            self.dens_resp_func_recip = js.fourier_inv(self.dens_resp_func, self.spat_coord2)
        self.dens_resp_func_recip = js.fourier_inv(self.dens_resp_func, self.spat_coord2)

    def add_dielectric_resp(self):
        """Add the dielectric response"""
        if self.dens_resp_func_recip == []:
            self.add_recip_dens_resp_func()
            self.dielectric_resp = js.epsilon(self.dens_resp_func_recip, np.real(tools.inv_rev_vec(self.qvec)), self.q_paral)
        else:
            self.dielectric_resp = js.epsilon(self.dens_resp_func_recip, np.real(tools.inv_rev_vec(self.qvec)), self.q_paral)
   
    def add_eigen_values(self):
        """Computes the eigen values and vector of the dielectric response"""
        if self.dielectric_resp == []:
            self.add_dielectric_resp()
            self.eig_v_eps, self.eig_rvec_eps = np.linalg.eig(self.dielectric_resp)
        else:
            self.eig_v_eps, self.eig_rvec_eps = np.linalg.eig(self.dielectric_resp)
    
    def add_weight(self):
        """Computes the weight associated with the different eigen values"""
        if self.eig_v_eps == []:
            self.add_eigen_values()
            self.weight, self.eig_v_eps = ts.weight_majerus(self.dielectric_resp)
        else:
            self.weight, self.eig_v_eps = ts.weight_majerus(self.dielectric_resp)
        
    def add_loss_func(self):
        """Computes the loss function of the computed slab"""
        self.loss_func = ts.loss_func_majerus(self.weight, self.eig_v_eps)
    
    
