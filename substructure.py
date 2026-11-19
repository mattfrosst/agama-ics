"""
Dark-matter substructure for AGAMA disc + halo models (run_AGAMA_model.py --sub-frac ...).

The host halo density rho_host(r) (from [Potential dark halo]) is split into
    rho_host(r) = rho_smooth(r) + <rho_sub>(r)
where <rho_sub>(r) is the azimuthally averaged density of a population of bound subhaloes.
The smooth part gets a QuasiSpherical DF in the total potential as before, and <rho_sub> is
included as a static spherical component of the self-consistent model, so the mean potential
(and hence the disc's equilibrium) is the same as without substructure.

Subhalo population
  * mass function   dN/dm ~ m^alpha, alpha = -1.9 (Springel et al. 2008, Aquarius),
                    between m_min and m_max; total mass = f_sub * M_host (realised by sampling).
  * radial profile  <rho_sub>(r) = f_max * r/(r + r_b) * rho_host(r) * exp(-(r/r_cut)^2)
                    i.e. anti-biased (subhaloes avoid the centre, as in simulations); the local
                    subhalo fraction r/(r+r_b) rises outward and never exceeds f_max < 1, so
                    rho_smooth stays positive. r_cut defaults to r200 of the host.
  * orbits          centres sampled from an isotropic QuasiSpherical DF of <rho_sub> in the
                    total potential, i.e. the subhalo system is itself in equilibrium.
  * internal        each subhalo is a truncated NFW sphere (AGAMA Spheroid gamma=1,beta=3,alpha=1)
                    with scale radius r_s = r200(m)/c(m), c(m) = c0 (m / 1e12 Msun)^-0.1, and an
                    exp(-(r/r_t)^2) cutoff at its Jacobi radius r_t = r (m / 3 M(<r))^(1/3) at the
                    initial orbital radius; 'm' is the bound mass after truncation.
                    Particles are sampled from its own isotropic DF (isolated equilibrium).
  * particles       equal mass to the smooth-halo particles (avoids mass-segregation heating).

Units: Msun, kpc, km/s.
"""
import numpy as np
import agama

G = 4.30091727067736e-06


def r200_of_mass(m, H=0.07):
    """r200 [kpc] for mass m [Msun], H in km/s/kpc (0.07 = 70 km/s/Mpc)."""
    return (G * m / (100 * H ** 2)) ** (1. / 3)


class SubhaloPopulation:
    def __init__(self, dens_host, M_host, f_sub=0.05, m_min=None, m_max=None, slope=-1.9,
                 r_b=None, r_cut=None, c0=10.0, particle_mass=None, rng=None, nmin_warn=100):
        self.dens_host = dens_host
        self.M_host = float(M_host)
        self.f_sub = float(f_sub)
        self.mp = float(particle_mass)
        self.m_min = float(m_min) if m_min else 100 * self.mp
        self.m_max = float(m_max) if m_max else 0.01 * self.M_host
        self.slope = slope
        self.r200 = r200_of_mass(self.M_host)
        self.r_cut = float(r_cut) if r_cut else self.r200
        self.r_b = float(r_b) if r_b else 0.5 * self.r200
        self.c0 = c0
        self.rng = rng or np.random.default_rng()
        self.nmin_warn = nmin_warn
        if self.m_min >= self.m_max:
            raise ValueError('sub m_min (%.3g) must be < m_max (%.3g)' % (self.m_min, self.m_max))
        self._draw_masses()
        self._build_densities()

    # ---- mass function -------------------------------------------------------------------
    def _draw_masses(self):
        a1 = self.slope + 1
        target = self.f_sub * self.M_host
        masses = []
        while sum(masses) < target:  # inverse-CDF sampling of a truncated power law
            u = self.rng.random()
            m = (self.m_min ** a1 + u * (self.m_max ** a1 - self.m_min ** a1)) ** (1 / a1)
            masses.append(m)
        # drop the last one if that brings the total closer to the target
        if len(masses) > 1 and abs(sum(masses[:-1]) - target) < abs(sum(masses) - target):
            masses.pop()
        # equal-mass particles: round each subhalo to an integer number of particles
        npart = np.maximum(1, np.round(np.array(masses) / self.mp).astype(int))
        order = np.argsort(-npart)
        self.npart = npart[order]
        self.mass = self.npart * self.mp
        self.M_sub = self.mass.sum()
        self.f_real = self.M_sub / self.M_host

    # ---- mean subhalo density and smooth halo --------------------------------------------
    def _shape(self, r):
        return r / (r + self.r_b) * np.exp(-(r / self.r_cut) ** 2)

    def _build_densities(self):
        # normalise f_max so that int <rho_sub> dV = M_sub
        r = np.logspace(-3, np.log10(10 * self.r_cut), 4000)
        rho_h = self.dens_host.density(np.column_stack((r, 0 * r, 0 * r)))
        integrand = 4 * np.pi * r ** 2 * rho_h * self._shape(r)
        self.f_max = self.M_sub / np.trapezoid(integrand, r)
        if self.f_max >= 1:
            raise ValueError('f_sub=%.3g needs a local subhalo fraction f_max=%.2f >= 1; lower --sub-frac '
                             'or --sub-rb' % (self.f_real, self.f_max))
        fmax, shape, host = self.f_max, self._shape, self.dens_host
        grid = dict(gridSizeR=60, rmin=1e-2, rmax=10 * self.r_cut, lmax=0, symmetry='s')

        def rho_sub(x):
            rr = np.sqrt(np.sum(np.atleast_2d(x) ** 2, axis=1))
            return fmax * shape(rr) * host.density(np.atleast_2d(x))

        def rho_smooth(x):
            rr = np.sqrt(np.sum(np.atleast_2d(x) ** 2, axis=1))
            return (1 - fmax * shape(rr)) * host.density(np.atleast_2d(x))

        self.dens_mean = agama.Density(type='DensitySphericalHarmonic', density=rho_sub, **grid)
        self.dens_smooth = agama.Density(type='DensitySphericalHarmonic', density=rho_smooth, **grid)

    def summary(self):
        return ('%d subhaloes, M_sub=%.4g Msun (f_sub=%.4f; requested %.4f), m=[%.3g, %.3g] Msun, '
                'N_part/subhalo=[%d, %d], local fraction f_max=%.3f, r_b=%.1f, r_cut=%.1f kpc' %
                (len(self.mass), self.M_sub, self.f_real, self.f_sub, self.mass.min(), self.mass.max(),
                 self.npart.min(), self.npart.max(), self.f_max, self.r_b, self.r_cut))

    # ---- orbits and internal structure ---------------------------------------------------
    def sample(self, pot, softening=None):
        """Sample subhalo centres in total potential `pot` and particles for each subhalo.
        Returns dict with particle arrays and a catalogue."""
        nsub = len(self.mass)
        dfc = agama.DistributionFunction(type='QuasiSpherical', potential=pot, density=self.dens_mean)
        centres, _ = agama.GalaxyModel(potential=pot, df=dfc).sample(nsub)
        rc = np.linalg.norm(centres[:, :3], axis=1)
        # enclosed total mass for the Jacobi radius, from the circular velocity of the total potential
        vc2 = -rc * np.sum(pot.force(centres[:, :3]) * centres[:, :3], axis=1) / rc
        Menc = vc2 * rc / G
        r_t = rc * (self.mass / (3 * Menc)) ** (1. / 3)
        cm = self.c0 * (self.mass / 1e12) ** -0.1
        r_s = r200_of_mass(self.mass) / cm

        pos, vel, sid = [], [], []
        for i in range(nsub):
            sp = agama.Potential(type='Spheroid', mass=self.mass[i], gamma=1, beta=3, alpha=1,
                                 scaleRadius=r_s[i], outerCutoffRadius=r_t[i], cutoffStrength=2)
            df = agama.DistributionFunction(type='QuasiSpherical', potential=sp, density=sp)
            xv, _ = agama.GalaxyModel(potential=sp, df=df).sample(int(self.npart[i]))
            xv[:, :3] -= xv[:, :3].mean(axis=0); xv[:, 3:] -= xv[:, 3:].mean(axis=0)  # exact centring
            pos.append(xv[:, :3] + centres[i, :3]); vel.append(xv[:, 3:] + centres[i, 3:])
            sid.append(np.full(len(xv), i))
        cat = np.column_stack((np.arange(nsub), self.mass, self.npart, centres, rc, r_s, r_t, cm))
        warn = []
        if self.npart.min() < self.nmin_warn:
            warn.append('%d subhaloes have < %d particles' % ((self.npart < self.nmin_warn).sum(), self.nmin_warn))
        if softening and (r_t < 2 * softening).any():
            warn.append('%d subhaloes have tidal radius < 2x softening (%.3g kpc)' %
                        ((r_t < 2 * softening).sum(), softening))
        return dict(pos=np.concatenate(pos), vel=np.concatenate(vel),
                    mass=np.full(sum(len(p) for p in pos), self.mp), subid=np.concatenate(sid),
                    catalogue=cat, warnings=warn)

    CAT_HEADER = 'id\tmass[Msun]\tnpart\tx\ty\tz\tvx\tvy\tvz\tr[kpc]\tr_s[kpc]\tr_t[kpc]\tc200'
