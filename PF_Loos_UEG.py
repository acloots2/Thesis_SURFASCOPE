import math
from math import pi
import numpy as np
import numba
from numba import jit
from scipy.special import zeta

def spinpol(dens_up, dens_down = 0):
    if dens_down == 0:
        return 0
    else:
        return (dens_up-dens_down)/(dens_up+dens_down)
    
def density_from_radius(radius, dimension = 3):
    if dimension == 3:
        return 3/(4*math.pi*radius**3)
    elif dimension == 2:
        return 1/(math.pi*radius**2)
    else:
        return 1/(2*radius)

def radius_from_density(density, dimension = 3):
    print(density)
    if type(density) == float or type(density) == int or type(density) == complex:
        if dimension == 3:
            return np.power(3/(4*pi)*np.power(density, -1), 1/3)
        elif dimension == 2:
            return np.power(1/pi*np.power(density, -1), 1/2)
        else:
            return 1/(2*density)  
    else:
        npnt = len(density)
        radius = np.zeros(npnt)
        if dimension == 3:
            for i in range(npnt):
                if density[i] != 0:
                    radius[i] = (3/(4*pi)*density[i]**(-1))**(1/3)
        elif dimension == 2:
            for i in range(npnt):
                if density[i] != 0:
                    radius[i] = (1/pi*density[i]**(-1))**(1/2)
        else:
            for i in range(npnt):
                if density[i] != 0:
                    radius = 1/(2*density[i])  
        return radius
        

    
def fermi_wavevector(radius, dimension = 3):
    npnt = len(radius)
    k_f = np.zeros(npnt)
    if dimension == 3:
        for i in range(npnt):
            if radius[i] != 0:
                k_f[i] = (9*math.pi/4)**(1/3)/radius[i]
    elif dimension == 2:
        for i in range(npnt):
            if radius[i] != 0:
                k_f[i] = np.sqrt(2)/radius[i]
    else:
        for i in range(npnt):
            if radius[i] != 0:
                k_f[i] = math.pi/4/radius[i]
    return k_f
    
def kin_energy(radius, dimension = 3, spinpol_var = 0):
    k_f = fermi_wavevector(radius, dimension)
    spin_fac = ((1+spinpol_var)**((dimension+2)/dimension)+(1-spinpol_var)**((dimension+2)/dimension))/2
    dim_fac = dimension/(2*(dimension+2))
    fac = spin_fac*dim_fac
    return fac*np.power(k_f, 2)

def x_energy(radius, dimension = 3, spinpol_var = 0):
    k_f = fermi_wavevector(radius, dimension)
    spin_fac = ((1+spinpol_var)**((dimension+1)/dimension)+(1-spinpol_var)**((dimension+1)/dimension))/2
    return -2*dimension/(math.pi*(dimension**2-1))*k_f*spin_fac

def c_energy(radius, dimension = 3, spinpol_var = 0):
    if radius.any() < 1:
        if spinpol_var == 0:
            if dimension == 3:
                A = (1-np.log(2))/math.pi**2
                B = -0.07111
                C = np.log(2)/6-3/(4*math.pi**2)*zeta(3)
                D = (9*math.pi/4)**(1/3)*(math.pi**2-6)/(24*math.pi**3)
                E = (9*math.pi/4)**(1/3)*(math.pi**2-12*np.log(2))/(4*math.pi**3)
                F = -0.01
            elif dimension == 2:
                A = 0
                B = np.log(2)-1
                C = dirichlet(2)-8/math.pi**2*dirichlet(4)
                D = -np.sqrt(2)*(10/(3*math.pi)-1)
                E = 0
                F = 0
                
            else:
                A = 0
                B = -math.pi**2/360
                C = 0
                D = 0
                E = 0
                F = 0.008446
        else:
            if dimension == 3:
                A = (1-np.log(2))/(2*math.pi**2)
                B = -0.049917
                C = np.log(2)/6-3/(4*math.pi**2)*zeta(3)
                D = 1/2**(7/3)*(9*math.pi/4)**(1/3)*(math.pi**2+6)/(24*math.pi**3)
                E = 1/2**(4/3)*(9*math.pi/4)**(1/3)*(math.pi**2-12*np.log(2))/(4*math.pi**3)
                F = 0
            elif dimension == 2:
                A = 0
                B = (np.log(2)-1)/2
                C = dirichlet(2)-8/math.pi**2*dirichlet(4)
                D = -1/4*(10/(3*math.pi)-1)
                E = 0
                F = 0
            else:
                A = 0
                B = -math.pi**2/360
                C = 0
                D = 0
                E = 0
                F = 0.008446
        return A*np.log(radius)+B+C+(D+E)*radius*np.log(radius)+F*radius
    elif radius.any() > 200:
        #Might add the value for different lattices
        if dimension  == 3:
            return -0.895930/radius+1.325/radius**(3/2)-0.365/radius**2
        elif dimension == 2:
            return -1.106103/radius+0.795/radius**(3/2)
        else:
            return (0.577215664-np.log(2))/2/radius+0.359933/radius**(3/2)
    elif 100<radius.any()<120:
        if dimension == 3:
            return -0.895930/radius+1.3379/radius**(3/2)-0.55270/radius**2
        elif dimension == 2:
            return -1.106103/radius+0.814/radius**(3/2)+0.266297/radius**2-2.63286/radius**(5/2)+6.246358/radius**3
        else:
            raise ValueError("No xc-energy available for the 1D case in this regime")
    else:
        if dimension == 3:
            if spinpol_var ==0:
                A = -0.214488
                B = 1.68634
                C = 0.49053
            else:
                A = -0.09399
                B = 1.5268
                C = 0.28882
            return A/(1+B*radius**(1/2)+C*radius)
        elif dimension == 2:
            if spinpol_var == 0:
                a0 = 0.1863052
                a1 = 6.821839
                a2 = 0.155226 
                a3 = 3.423013
            else:
                a0 = -0.2909102
                a1 = -0.6243836
                a2 = 1.656628
                a3 = 3.791685
            A = 2*(a1+2*a2)/(2*a1*a2-a3-a1**2)
            B = 1/a1-1/(a1+2*a2)
            C = a1/a3-2*a2/a3+1/(a1+2*a2)
            F = 1+(2*a2-a1)*(1/(a1+2*a2)-2*a2/a3)
            D = (F-a2*C)/np.sqrt(a3-a2**2)
            return a0*(1+A*radius*(B*np.log((np.sqrt(radius)+a1)/np.sqrt(radius))+C/2*np.log((radius+2*a2*np.sqrt(radius)+a3)/radius)+D*(np.arctan((np.sqrt(radius)+a2)/np.sqrt(a3-a2**2))-pi/2)))
        else:
            t = (np.sqrt(1+4*0.414254*radius)-1)/(2*0.414254*radius)
            k = 0.414254
            c0 = 0.414254*(0.577215664-np.log(2))/2
            c1 = 4*k*(0.577215664-np.log(2))/2+k**(3/2)*0.359933
            c2 = 5*-math.pi**2/360+0.008446/k
            c3 = 0.008446
            return t**2*(c0*t**0*(1-t)**3+c1*t*(1-t)**2+c2*t**2*(1-t)+c3*t**3)


def e_total(dens, dimension, spinpol_var = 0):
    rad = radius_from_density(dens, dimension)
    npnt = len(rad)
    index = np.argwhere(rad)[:, 0]
    rad_no_zeros = rad[index]
    e_total = np.zeros(npnt)
    if rad.any()>120:
        e_total[index] = c_energy(rad_no_zeros, dimension, spinpol_var)
    else:
        e_total[index] = kin_energy(rad_no_zeros, dimension, spinpol_var)+x_energy(rad_no_zeros, dimension, spinpol_var)+c_energy(rad_no_zeros, dimension, spinpol_var)
    return e_total
            
        
def x_energy_lda(radius):
    return -3/4*(3/2*math.pi)**(2/3)/radius



    



def dirichlet(s):
    sum_var = 0
    for i in range(100):
        sum_var += (-1)**i/(2*i+1)**s
    return sum_var
