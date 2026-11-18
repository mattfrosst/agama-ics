import numpy as np

from scipy.special   import kn, iv
from scipy.special   import lambertw
from scipy.integrate import quad
from scipy.integrate import dblquad
from scipy.optimize  import brentq

import matplotlib.pylab as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

########################################################################
G = 4.30091727067736e-06
########################################################################
def hernq_vr(r, M, a):
    v_H = np.sqrt(G * (M * r**2 / (r + a)**2) / r)
    return v_H

def hernq_potential(r, M, a):
    return -G * M / (r + a)

def hernq_potential_derivative(r, M, a):
    return G * M / (r + a)**2

def hernq_potential_second_derivative(r, M, a):
    return - 2 * G * M / (r + a)**3

def disk_surface_density(r, M, Rd):
    Sigma_0 = M / (2 * np.pi * Rd ** 2)
    Sigma_R = Sigma_0 * np.exp(-r / Rd)
    return Sigma_R

def disk_potential(r, M, Rd):
    y = r / (2 * Rd)
    Phi  = (G * M) / (2 * Rd**2) * r * (iv(1, y) * kn(0, y) - iv(0, y) * kn(1, y))
    return Phi

def disk_potential_derivative(r, M, Rd):
    y = r / (2 * Rd)
    dPhi_dR = (G * M * r / (2 * Rd** 3)) * (iv(0, y) * kn(0, y) - iv(1, y) * kn(1, y))
    return dPhi_dR

def disk_potential_second_derivative(r, M, Rd):
    y = r / (2 * Rd)
    d2Phi_dR2 = (G * M) / (2 * Rd**4) * (iv(1,y) * (r*kn(0,y)+Rd*kn(1,y)) + iv(0,y) * (Rd*kn(0,y)-r*kn(1,y)))
    return d2Phi_dR2

def total_potential(r, Mdisk, Rd, Mhalo, a):
    return disk_potential(r, Mdisk, Rd) + hernq_potential(r, Mhalo, a)

def total_potential_derivative(r, Mdisk, Rd, Mhalo, a):
    #derivative of a sum is the sum of derivatives
    # f(x) = g(x) + h(x) => f'(x) = g'(x) + h'(x)
    return disk_potential_derivative(r, Mdisk, Rd) + hernq_potential_derivative(r, Mhalo, a)

def total_potential_second_derivative(r, Mdisk, Rd, Mhalo, a):
    #2nd derivative of a sum is the sum of 2nd derivatives
    # f(x) = g(x) + h(x) => f''(x) = g''(x) + h''(x)
    return disk_potential_second_derivative(r, Mdisk, Rd) + hernq_potential_second_derivative(r, Mhalo, a)

def set_veldispersions(rad, M_disk, Rd, M_halo, a_halo):
    '''
    The calculation being performed here partially follows
    the prescription found in Springel & White, 1999.
    '''
    #disk - Eqn. 29, 30, 31 => Epicycle approximation
    kappa2 = 3/rad * total_potential_derivative(rad,M_disk,Rd,M_halo,a_halo) + total_potential_second_derivative(rad, M_disk, Rd, M_halo, a_halo)
    gamma2 = 4/(kappa2*rad) * total_potential_derivative(rad,M_disk,Rd,M_halo,a_halo)

    #Force Q = 1.2 = sigR * kappa / (3.36 * G * Sigma)
    Q = 1.2
    sSigmas  = disk_surface_density(rad,M_disk,Rd/1.5)
    sRs      = (Q * 3.36 * G * sSigmas) / np.sqrt(kappa2) #set Q = 1.05, populate sigma R array (3.528 = 1.05 * 3.36)
    return sRs

def work_out_sigmar0(M_disk, C):
    Rd = 2.0 #kpc
    Rh = C * Rd #kpc
    M_halo = 1e12 #Msol
    
    sigmar0 = set_veldispersions(0.1, M_disk, Rd, M_halo, Rh)
    return sigmar0
    
def work_out_all_inputs(MD_in, C_in, sigma_in,  save=False):
    '''Makes Gadget param.txt and Submit.sh files
    '''
    model_name = 'C%s_MD%s'%(str(round(C_in,3)).replace('.','p'),str(round(MD_in,3)).replace('.','p'))

    C_str = str(round(C_in,3)).replace('.','p')
    MD_str = str(round(MD_in,3)).replace('.','p')

    Mdisk = MD_in * 1.0e12
    Chalo = 2.0 * C_in
    sigmar0 = work_out_sigmar0(Mdisk, Chalo)
    print(sigmar0, sigma_in)
        
    #make param_file
    param_file = '''
    ### All parameters listed here are in dimensional units:  [M] = Msun, [R] = kpc, [v] = km/s

    #initial potentila is the best-fit potential from McMillan(2017)

    [Potential thin disk]
    type = Disk
    # total disk mass [Msun]
    mass = {1}
    # scale radius of exponential disk [kpc]
    scaleRadius = 2.0
    # scale height [kpc], positive value corresponds to exponential vertical profile, and negative - to isothermal (sech^2)
    scaleHeight = -0.2

    [Potential dark halo]
    #Hernquist profile
    type = Dehnen
    # total halo mass [Msun]
    mass = 1.0e+12
    # inner power-law slope
    gamma = 1
    # break radius in density profile [kpc] | C = Rh/Rd = 10
    scaleRadius = {0}
    # exponential cutoff radius [kpc]
    outerCutoffRadius = 200.0

    ### parameters of thin disk distribution function
    [DF thin disk]
    type    = QuasiIsothermal
    # surface density normalization (value at origin)  [Msun/kpc^2]
    mass = {1}
    # scale radius of the (exponential) disk surface density [kpc]
    Rdisk   = 2.0
    # scale height [kpc]
    Hdisk   = 0.2
    ### parameters describing the dependence of DF on the azimuthal, radial and vertical actions
    # normalization of radial velocity dispersion at R=0 [km/s]
    sigmar0 = {2}
    # lower limit on the radial velocity dispersion: take max(sigmar,sigmamin) [km/s]
    sigmamin= 1.0
    # scale radius of radial velocity dispersion: sigmar=sigmar0*exp(-R/Rsigmar); default should be 2*Rdisk [kpc]
    Rsigmar = 4.0

    ### parameters of the halo component of self-consistent model
    [SelfConsistentModel halo]
    # definition of spherical grid for computing the density profile:
    # min and max radii [kpc], grid is in log-radius
    rminSph        = 0.05
    rmaxSph        = 5000.
    # number of radial grid points
    sizeRadialSph  = 50
    # order of spherical-harmonic expansion
    # (lmax, must be even; the number of 'beams' (different values of theta) is lmax/2+1
    lmaxAngularSph = 4

    ### parameters of the disk+stellar halo component of self-consistent model
    [SelfConsistentModel disk]
    # definition of grid in cylindrical radius - radii in kpc
    RminCyl        = 0.05
    RmaxCyl        = 50
    sizeRadialCyl  = 50
    # definition of grid in vertical direction
    zminCyl        = 0.05
    zmaxCyl        = 5.0
    sizeVerticalCyl= 16

    ### parameters for the potential solvers for the entire self-consistent model
    [SelfConsistentModel]
    # definition of spherical grid for constructing multipole potential expansion;
    # this grid should encompass that of the halo component, but also should resolve
    # the bulge density profile at small radii; meaning of parameters is the same
    rminSph        = 0.01
    rmaxSph        = 5000.
    sizeRadialSph  = 50
    lmaxAngularSph = 4
    # definition of cylindrical grid for CylSpline expansion;
    # again it should be larger and more detailed than the grid used for computing
    # density profiles, and should resolve the thin gas disk
    RminCyl        = 0.01
    RmaxCyl        = 50
    sizeRadialCyl  = 50
    zminCyl        = 0.04
    zmaxCyl        = 10
    sizeVerticalCyl= 30
    # set to true to speed up the computation at the expense of a somewhat lower accuracy
    useActionInterpolation = False

    ### Parameters for fitting the model to data - not yet used
    [Data]
    # used only for outputting the velocity distribution
    SolarRadius = 2.0    
    '''.format(Chalo,Mdisk, sigmar0)
    
    if save:
        with open('./AGAMA_'+ model_name +'.ini', 'w') as model:
            model.write(param_file)
    
    return

if __name__ == '__main__':
    sigma = [70, 90, 110, 120, 130]
    MD = [0.01, 0.02, 0.03, 0.04, 0.05]
    C = [5.0, 7.5, 10.0, 15.0, 20.0]
    for i in range(len(MD)):
        for j in range(len(C)):
            work_out_all_inputs(MD[i], C[j], sigma[i], save=True)
    pass

