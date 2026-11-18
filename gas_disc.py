"""
Isothermal gas disc in vertical hydrostatic equilibrium, for use with AGAMA
self-consistent models (run_AGAMA_model.py --gas-frac ...).

Model (cf. Springel, Di Matteo & Hernquist 2005; Wang et al. 2010):
  * surface density      Sigma_g(R) = M_g / (2 pi R_g^2) exp(-R/R_g)   (truncated at R_trunc)
  * vertical structure   rho_g(R,z) = rho_0(R) exp( -[Phi(R,z) - Phi(R,0)] / c_s^2 )
    with rho_0(R) fixed by  int rho_g dz = Sigma_g(R).  Phi is the TOTAL potential
    (stars + halo + gas), so the gas self-gravity is included by iterating together
    with the stellar/halo DFs in the SelfConsistentModel.
  * rotation             v_phi^2 = R dPhi/dR + c_s^2 R dln(rho_g)/dR   (pressure support),
    v_R = v_z = 0 (no random motions; support comes from pressure).
  * internal energy      u = c_s^2               for an isothermal EOS (Gadget-4 ISOTHERM_EQS)
                         u = c_s^2 / (gamma - 1) for an adiabatic EOS with index gamma.

Units: Msun, kpc, km/s.
"""
import numpy as np
import agama


class GasDisc:
    def __init__(self, mass, Rdisk, cs, Rtrunc=None, zmax=10.0, nR=300, nz=400,
                 grid_zmin=0.002, grid_sizeR=60, grid_sizez=50):
        self.mass = float(mass)
        self.Rd = float(Rdisk)
        self.cs2 = float(cs) ** 2
        self.Rtrunc = float(Rtrunc) if Rtrunc else 12.0 * self.Rd
        self.zmax = float(zmax)
        # tabulation grids for rho_0(R): log-spaced in R, and in z from ~0 to zmax
        self.Rtab = np.logspace(np.log10(1e-3 * self.Rd), np.log10(self.Rtrunc), nR)
        self.ztab = np.r_[0.0, np.logspace(-5, np.log10(self.zmax), nz - 1)]
        self.grid = dict(gridSizeR=grid_sizeR, gridSizez=grid_sizez, Rmin=1e-2 * self.Rd,
                         Rmax=self.Rtrunc, zmin=grid_zmin, zmax=self.zmax, mmax=0, symmetry='a')
        self.pot = None
        self.lnrho0 = None
        # normalisation of the (truncated) exponential so that the total mass is exactly `mass`
        x = self.Rtrunc / self.Rd
        self.Sigma0 = self.mass / (2 * np.pi * self.Rd ** 2 * (1 - np.exp(-x) * (1 + x)))

    def Sigma(self, R):
        return np.where(R <= self.Rtrunc, self.Sigma0 * np.exp(-R / self.Rd), 0.0)

    def initial_density(self, h=0.05):
        """sech^2 starting guess (before any potential is available)."""
        Sig0, Rd, Rt = self.Sigma0, self.Rd, self.Rtrunc

        def rho(x):
            R = np.hypot(x[:, 0], x[:, 1]); z = x[:, 2]
            return np.where(R <= Rt, Sig0 * np.exp(-R / Rd) / (2 * h) / np.cosh(z / h) ** 2, 0.0)
        return agama.Density(type='DensityAzimuthalHarmonic', density=rho, **self.grid)

    def update(self, pot):
        """Recompute rho_0(R) for hydrostatic equilibrium in potential `pot`; return AGAMA Density."""
        self.pot = pot
        R, z = self.Rtab, self.ztab
        RR = np.repeat(R, len(z)); ZZ = np.tile(z, len(R))
        Phi = pot.potential(np.column_stack((RR, 0 * RR, ZZ))).reshape(len(R), len(z))
        f = np.exp(-(Phi - Phi[:, :1]) / self.cs2)
        I = 2 * np.trapezoid(f, z, axis=1)  # = int_{-zmax}^{zmax} exp(-dPhi/cs^2) dz
        self.lnrho0 = np.log(self.Sigma(R) + 1e-300) - np.log(I)
        self.thickness = I / 2  # equivalent half-thickness Sigma/(2 rho_0)
        return agama.Density(type='DensityAzimuthalHarmonic', density=self.rho, **self.grid)

    def rho(self, x):
        x = np.atleast_2d(x)
        R = np.hypot(x[:, 0], x[:, 1]); z = np.abs(x[:, 2])
        lnrho0 = np.interp(np.log(np.maximum(R, self.Rtab[0])), np.log(self.Rtab), self.lnrho0)
        dPhi = (self.pot.potential(np.column_stack((R, 0 * R, z))) -
                self.pot.potential(np.column_stack((R, 0 * R, 0 * R))))
        out = np.exp(lnrho0 - dPhi / self.cs2)
        return np.where((R <= self.Rtrunc) & (z <= self.zmax), out, 0.0)

    def sample(self, n, density, gamma=None):
        """Sample n particles. Returns (posvel[n,6], mass[n], u[n])."""
        pos, mass = density.sample(n)
        mass = np.full(n, self.mass / n)  # equal-mass particles with the exact total mass
        R = np.hypot(pos[:, 0], pos[:, 1]); z = pos[:, 2]
        # radial gravity at the particle position and pressure-gradient correction
        dPhidR = -(pos[:, 0] * self.pot.force(pos)[:, 0] + pos[:, 1] * self.pot.force(pos)[:, 1]) / np.maximum(R, 1e-12)
        dR = np.maximum(1e-3 * R, 1e-4)
        xp = np.column_stack((R + dR, 0 * R, z)); xm = np.column_stack((np.maximum(R - dR, 1e-6), 0 * R, z))
        dlnrho = (np.log(self.rho(xp) + 1e-300) - np.log(self.rho(xm) + 1e-300)) / (xp[:, 0] - xm[:, 0])
        vphi2 = R * dPhidR + self.cs2 * R * dlnrho
        vphi = np.sqrt(np.maximum(vphi2, 0.0))
        phi = np.arctan2(pos[:, 1], pos[:, 0])
        vel = np.column_stack((-vphi * np.sin(phi), vphi * np.cos(phi), 0 * R))
        u = np.full(n, self.cs2 if gamma is None else self.cs2 / (gamma - 1))
        return np.column_stack((pos, vel)), mass, u, np.mean(vphi2 < 0)
