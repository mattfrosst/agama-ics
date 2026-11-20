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


def disc_kinematics(Mdisk, a_halo, Q_target, sigz_sigR, Rd=2.0, Mhalo=1.0e12, R_Q=None, hz_Rd=None):
    """Analytic first guess of the disc velocity-dispersion parameters for a target Toomre Q.

    Q is imposed at R_Q (default: the disc scale radius Rd) using the epicyclic approximation
    in the razor-thin exponential disc + Hernquist potential (Springel & White 1999, eqs 29-31):
        kappa^2 = 3/R dPhi/dR + d2Phi/dR2,   sigma_R(R_Q) = Q * 3.36 G Sigma(R_Q) / kappa(R_Q).
    Both dispersions decline as exp(-R / 2Rd), so sigma_z/sigma_R = sigz_sigR at all radii:
        sigmar0 = sigma_R(R_Q) exp(R_Q / 2Rd),   sigmaz0 = sigz_sigR * sigmar0.
    The disc thickness is then an output; its value at R_Q is estimated for the initial-guess
    density with a self-gravitating isothermal sheet, rho ~ sech^2(z / 2h),
    h = sigma_z^2 / (2 pi G Sigma) (an overestimate where the halo adds vertical gravity).
    run_AGAMA_model.py then rescales sigmar0 and sigmaz0 together so that the self-consistent
    model has exactly Q_target at R_Q (see the [Calibration] section of the ini file).

    If hz_Rd is given, the disc thickness is a target instead and sigz_sigR is not used:
    hz_Rd = exponential scale height h_z / Rd at R_Q, where rho ~ sech^2(z / 2 h_z), i.e. the
    usual observational scale height (rho ~ exp(-|z|/h_z) away from the midplane). sigma_z is
    first estimated for a self-gravitating sheet, sigma_z^2 = 2 pi G Sigma h_z (an underestimate
    where the halo adds vertical gravity), and calibrated in run_AGAMA_model.py; sigma_z/sigma_R
    then follows from the two targets."""
    R_Q = Rd if R_Q is None else R_Q
    kappa2 = 3 / R_Q * total_potential_derivative(R_Q, Mdisk, Rd, Mhalo, a_halo) + \
        total_potential_second_derivative(R_Q, Mdisk, Rd, Mhalo, a_halo)
    Sigma = disk_surface_density(R_Q, Mdisk, Rd)
    sigR = Q_target * 3.36 * G * Sigma / np.sqrt(kappa2)
    Rsig = 2.0 * Rd
    sigmar0 = sigR * np.exp(R_Q / Rsig)
    sigmaz0 = sigz_sigR * sigmar0
    if hz_Rd is not None:
        h = hz_Rd * Rd
        sigmaz0 = np.sqrt(2 * np.pi * G * Sigma * h) * np.exp(R_Q / Rsig)
    else:
        h = (sigz_sigR * sigR) ** 2 / (2 * np.pi * G * Sigma)
    return dict(sigmar0=sigmar0, sigmaz0=sigmaz0, Rsigma=Rsig, h=h, sigR_RQ=sigR, R_Q=R_Q,
                kappa=np.sqrt(kappa2), Sigma=Sigma)


def fmt(x):
    return str(round(x, 3)).replace('.', 'p')


INI_TEMPLATE = """### All parameters listed here are in dimensional units:  [M] = Msun, [R] = kpc, [v] = km/s
### Model {name}: disc mass {Mdisk:.4g} Msun, halo scale radius {a_halo:g} kpc (C = {C:g}),
### {summary}

[Potential thin disk]
# initial guess only: the disc density is recomputed from the DF during the iterations
type = Disk
# total disk mass [Msun]
mass = {Mdisk}
# scale radius of exponential disk [kpc]
scaleRadius = {Rd}
# scale height [kpc]; negative = isothermal (sech^2) profile. Estimated from the target dispersions
scaleHeight = {mh:.4f}

[Potential dark halo]
# Hernquist profile
type = Dehnen
# total halo mass [Msun]
mass = 1.0e+12
# inner power-law slope
gamma = 1
# break radius in density profile [kpc] = C * Rd
scaleRadius = {a_halo}
# exponential cutoff radius [kpc] (ignored by run_AGAMA_model.py unless --halo-cutoff)
outerCutoffRadius = 200.0

### parameters of thin disk distribution function
[DF thin disk]
type    = QuasiIsothermal
# total disc mass [Msun]
mass = {Mdisk}
# scale radius of the (exponential) disk surface density [kpc]
Rdisk   = {Rd}
# radial velocity dispersion: sigma_R = sigmar0 exp(-R/Rsigmar) [km/s, kpc]
sigmar0 = {sigmar0:.4f}
Rsigmar = {Rsig}
# vertical velocity dispersion: sigma_z = sigmaz0 exp(-R/Rsigmaz) [km/s, kpc]
# (replaces Hdisk, which tied sigma_z to the local vertical frequency and gave sigma_z > sigma_R)
sigmaz0 = {sigmaz0:.4f}
Rsigmaz = {Rsig}
# lower limit on the velocity dispersions [km/s]
sigmamin= 1.0

### target disc structure, enforced by run_AGAMA_model.py
[Calibration]
# stellar Toomre Q at radius RQ [kpc] (rescales sigmar0)
Qtarget = {Q}
RQ = {Rd}
{calib_extra}

### parameters of the halo component of self-consistent model
[SelfConsistentModel halo]
# definition of spherical grid for computing the density profile:
# min and max radii [kpc], grid is in log-radius
rminSph        = 0.05
rmaxSph        = 5000.
# number of radial grid points
sizeRadialSph  = 50
# order of spherical-harmonic expansion
lmaxAngularSph = 4

### parameters of the disk component of self-consistent model
[SelfConsistentModel disk]
# definition of grid in cylindrical radius - radii in kpc
RminCyl        = 0.05
RmaxCyl        = 50
sizeRadialCyl  = 50
# definition of grid in vertical direction (first node set to resolve the disc thickness)
zminCyl        = {zmin:.4f}
zmaxCyl        = 5.0
sizeVerticalCyl= 20

### parameters for the potential solvers for the entire self-consistent model
[SelfConsistentModel]
rminSph        = 0.01
rmaxSph        = 5000.
sizeRadialSph  = 50
lmaxAngularSph = 4
RminCyl        = 0.01
RmaxCyl        = 50
sizeRadialCyl  = 50
zminCyl        = {zmin_pot:.4f}
zmaxCyl        = 10
sizeVerticalCyl= 30

[Data]
# used only for outputting the velocity distribution
SolarRadius = {Rd}
"""


def work_out_all_inputs(MD_in, C_in, Q_target=2.0, sigz_sigR=0.6, hz_Rd=None, Rd=2.0, save=False):
    """Writes AGAMA_C<C>_MD<MD>_Q<Q>.ini.
    MD_in: disc mass in units of the halo mass (1e12 Msun); C_in: halo scale radius / Rd;
    Q_target: stellar Toomre Q at R = Rd.
    hz_Rd: target exponential scale height h_z / Rd at Rd (then sigma_z/sigma_R follows), or
    None: constant sigma_z / sigma_R = sigz_sigR (then the thickness follows)."""
    model_name = 'C%s_MD%s_Q%s' % (fmt(C_in), fmt(MD_in), fmt(Q_target))
    if hz_Rd is not None:
        model_name += '_hz%s' % fmt(hz_Rd)
    Mdisk = MD_in * 1.0e12
    a_halo = C_in * Rd  # Hernquist scale radius [kpc]
    k = disc_kinematics(Mdisk, a_halo, Q_target, sigz_sigR, Rd=Rd, hz_Rd=hz_Rd)
    if hz_Rd is not None:
        summary = ('Toomre Q = %g and exponential scale height h_z = %g Rd at R = Rd '
                   '(sigma_z/sigma_R is an output)' % (Q_target, hz_Rd))
        calib_extra = ('# exponential scale height h_z / Rd at RQ, rho ~ sech^2(z / 2 h_z) (rescales sigmaz0)\n'
                       'hz_Rd = %g' % hz_Rd)
    else:
        summary = ('Toomre Q = %g at R = Rd, sigma_z/sigma_R = %g at all radii (disc thickness is an output)'
                   % (Q_target, sigz_sigR))
        calib_extra = ('# sigma_z / sigma_R (set by sigmaz0/sigmar0 with Rsigmaz = Rsigmar; reported only)\n'
                       'sigz_sigR = %g' % sigz_sigR)
    h0 = min(max(k['h'], 0.02), 1.0)
    print('%-22s sigma_R(Rd)=%6.2f  sigmar0=%6.2f  sigmaz0=%6.2f km/s  h_guess=%.3f kpc' %
          (model_name, k['sigR_RQ'], k['sigmar0'], k['sigmaz0'], h0))
    text = INI_TEMPLATE.format(name=model_name, Mdisk=Mdisk, a_halo=a_halo, C=C_in, Q=Q_target,
                               summary=summary, calib_extra=calib_extra, Rd=Rd, mh=-h0, sigmar0=k['sigmar0'],
                               sigmaz0=k['sigmaz0'], Rsig=k['Rsigma'],
                               zmin=min(0.05, h0 / 4), zmin_pot=min(0.04, h0 / 5))
    if save:
        with open('./AGAMA_' + model_name + '.ini', 'w') as f:
            f.write(text)
    return k


if __name__ == '__main__':
    Q_target = 2.0      # stellar Toomre Q at R = Rd (observed stellar discs: ~1.5-3)
    hz_Rd = 0.1         # exponential scale height h_z / Rd (edge-on galaxies: ~0.1-0.2);
                        # set to None to fix sigma_z/sigma_R = sigz_sigR instead
    sigz_sigR = 0.6     # sigma_z / sigma_R, used only if hz_Rd is None
    MD = [0.01, 0.02, 0.03, 0.04, 0.05]
    C = [5.0, 7.5, 10.0, 15.0, 20.0]
    for md in MD:
        for c in C:
            work_out_all_inputs(md, c, Q_target=Q_target, sigz_sigR=sigz_sigR, hz_Rd=hz_Rd, save=True)
