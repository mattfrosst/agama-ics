"""
Build a self-consistent disc + dark-halo equilibrium model with AGAMA
(Vasiliev 2019, MNRAS 482, 1525) from an .ini file written by make_AGAMA_ini.py,
and sample it into N-body particles.

Pipeline:
    make_AGAMA_ini.py   ->  AGAMA_<model>.ini
    run_AGAMA_model.py  ->  model_stars_final_<tag>.txt, model_dm_final_<tag>.txt
                            (+ optional mirrored Gadget HDF5 via write_mirror_particles.py)

Method (follows AGAMA's py/example_self_consistent_model*.py):
  1. Initial guess: disc and halo densities from the [Potential ...] sections.
  2. Disc DF: QuasiIsothermal ([DF thin disk]), rebuilt in the current potential.
     Halo DF: isotropic QuasiSpherical (Eddington-like) DF that reproduces the
     target halo density ([Potential dark halo]) in the current potential.
  3. Iterate: compute densities from DFs -> recompute total potential -> rebuild DFs.
  4. Sample particles from the final DFs in the final potential.
  Optional gas disc (--gas-frac > 0, see gas_disc.py): an isothermal exponential gas disc in
  vertical hydrostatic equilibrium, included as a static component of the self-consistent
  model and recomputed in the updated total potential after every iteration.
  Optional DM substructure (--sub-frac > 0, see substructure.py): a fraction of the halo mass is
  moved into bound NFW subhaloes on equilibrium orbits; their mean density is a static component
  of the model, so the total mean potential is unchanged. Subhalo particles are PartType3.
  Optional quiet start (--quiet-start, see quietstart.py): every component is sampled by
  stratification in action space (J_R, J_z, J_phi) from a weighted quasi-random pool, then
  point-mirrored; the text files then already contain the mirrored particles.

Units: Msun, kpc, km/s (G = 4.30092e-6 kpc (km/s)^2 / Msun). Output text files are
in these units; use --mass-unit when writing HDF5 to match Gadget's UnitMass.

Usage:
    python run_AGAMA_model.py AGAMA_C10p0_MD0p03.ini --ndisk 1000000 --nhalo 10000000
    python run_AGAMA_model.py AGAMA_*.ini --ndisk 200000 --nhalo 1000000 --hdf5
    python run_AGAMA_model.py AGAMA_C7p5_MD0p02.ini --gas-frac 0.1 --gas-cs 10 --ngas 20000 --hdf5
    python run_AGAMA_model.py AGAMA_C7p5_MD0p02.ini --sub-frac 0.05 --nhalo 1e7 --hdf5 --tag sub0p05
"""
import os
import sys
import time
import argparse
from configparser import RawConfigParser

import numpy as np
import agama

agama.setUnits(mass=1, length=1, velocity=1)  # Msun, kpc, km/s


def fix_halo_params(pot, halo_cutoff=False):
    """AGAMA >= 1.0 rejects outerCutoffRadius for type=Dehnen (older versions ignored it).
    Default: drop it -> plain Dehnen/Hernquist with the given total mass (as assumed by the
    analytic sigmar0 estimate in make_AGAMA_ini.py).
    halo_cutoff=True: use an equivalent Spheroid (alpha=1, beta=4, gamma) with the
    exponential cutoff exp(-(r/rcut)^2); note 'mass' is then the total mass AFTER truncation,
    which raises the inner density (by ~30% for a=20, rcut=200 kpc)."""
    pot = dict(pot)
    keys = {k.lower(): k for k in pot}
    if pot.get(keys.get('type', 'type'), '').lower() == 'dehnen' and 'outercutoffradius' in keys:
        if halo_cutoff:
            pot[keys['type']] = 'Spheroid'
            pot.setdefault('alpha', '1'); pot.setdefault('beta', '4')
            print('  halo: Dehnen + outerCutoffRadius -> truncated Spheroid (alpha=1, beta=4)')
        else:
            rc = pot.pop(keys['outercutoffradius'])
            print('  halo: ignoring outerCutoffRadius=%s for Dehnen profile (use --halo-cutoff to apply it)' % rc)
    return pot


def read_ini(filename, halo_cutoff=False):
    ini = RawConfigParser()
    ini.optionxform = str  # keep the case of parameter names
    if not ini.read(filename):
        raise IOError('Cannot read ini file %s' % filename)
    sec = lambda name: dict(ini.items(name)) if ini.has_section(name) else None
    cfg = dict(
        potDisk=sec('Potential thin disk'),
        potHalo=sec('Potential dark halo'),
        dfDisk=sec('DF thin disk'),
        scmHalo=sec('SelfConsistentModel halo'),
        scmDisk=sec('SelfConsistentModel disk'),
        scm=sec('SelfConsistentModel'),
        solarRadius=ini.getfloat('Data', 'SolarRadius') if ini.has_section('Data') else 8.0,
    )
    for key in ('potDisk', 'potHalo', 'dfDisk', 'scmHalo', 'scmDisk', 'scm'):
        if cfg[key] is None:
            raise ValueError('%s: missing section for %s' % (filename, key))
    # this option was removed from recent AGAMA versions; drop it silently
    for k in list(cfg['scm'].keys()):
        if k.lower() == 'useactioninterpolation':
            del cfg['scm'][k]
    cfg['potHalo'] = fix_halo_params(cfg['potHalo'], halo_cutoff)
    return cfg


def print_info(model, label, gas=None):
    densDisk = model.components[0].density
    densHalo = model.components[1].density
    pt0 = (2.0, 0, 0)
    pt1 = (2.0, 0, 0.25)
    print('%-6s Mdisk=%.4g Msun, Mhalo=%.4g Msun, Phi(0)=%.6g (km/s)^2, '
          'rho_disk(R=2,z=0)=%.4g, rho_disk(R=2,z=0.25)=%.4g, rho_halo(R=2)=%.4g Msun/kpc^3' %
          (label, densDisk.totalMass(), densHalo.totalMass(), model.potential.potential(0, 0, 0),
           densDisk.density(pt0), densDisk.density(pt1), densHalo.density(pt0)))
    if gas is not None and gas.lnrho0 is not None:
        h = np.interp([1.0, 2.0, 4.0, 8.0], gas.Rtab, gas.thickness)
        print('       Mgas=%.4g Msun, gas half-thickness Sigma/(2 rho0) at R=1,2,4,8 kpc: %s kpc' %
              (model.components[2].density.totalMass(), ', '.join('%.4f' % x for x in h)))
    sys.stdout.flush()


def build_model(cfg, niter=5, verbose=True, gas=None, sub=None):
    scm = dict(cfg['scm'])
    if gas is not None:
        # the gas layer is only ~10-50 pc thick: the CylSpline potential grid must resolve it
        keys = {k.lower(): k for k in scm}
        kz = keys.get('zmincyl', 'zminCyl')
        scm[kz] = min(float(scm.get(kz, 0.04)), gas.grid['zmin'] * 2.5)
        ks = keys.get('sizeverticalcyl', 'sizeVerticalCyl')
        scm[ks] = max(int(scm.get(ks, 30)), 40)
    model = agama.SelfConsistentModel(**scm)

    # initial guess for the density profiles
    densDiskInit = agama.Density(**cfg['potDisk'])
    # with substructure, the DF-based halo is only the smooth part of the host halo
    densHaloInit = sub.dens_smooth if sub is not None else agama.Density(**cfg['potHalo'])
    model.components.append(agama.Component(density=densDiskInit, disklike=True))
    model.components.append(agama.Component(density=densHaloInit, disklike=False))
    if gas is not None:
        model.components.append(agama.Component(density=gas.initial_density(), disklike=True))
    if sub is not None:  # static mean density of the subhalo population (always the last component)
        model.components.append(agama.Component(density=sub.dens_mean, disklike=False))
    model.iterate()
    if gas is not None:  # first hydrostatic update in the initial-guess potential
        gas.density = gas.update(model.potential)
        model.components[2] = agama.Component(density=gas.density, disklike=True)
        model.iterate()
    if verbose:
        print_info(model, 'init', gas)

    def make_dfs():
        dfDisk = agama.DistributionFunction(potential=model.potential, **cfg['dfDisk'])
        # isotropic DF reproducing the target halo density in the current total potential
        dfHalo = agama.DistributionFunction(type='QuasiSpherical', potential=model.potential,
                                            density=densHaloInit)
        return dfDisk, dfHalo

    for it in range(1, niter + 1):
        dfDisk, dfHalo = make_dfs()
        model.components[0] = agama.Component(df=dfDisk, disklike=True, **cfg['scmDisk'])
        model.components[1] = agama.Component(df=dfHalo, disklike=False, **cfg['scmHalo'])
        t0 = time.time()
        model.iterate()
        if gas is not None:  # re-establish hydrostatic equilibrium in the new potential
            gas.density = gas.update(model.potential)
            model.components[2] = agama.Component(density=gas.density, disklike=True)
            model.iterate()
        if verbose:
            print_info(model, 'it%d' % it, gas)
            print('       (iteration took %.1f s)' % (time.time() - t0))

    if gas is not None:  # gas profile consistent with the final potential
        gas.density = gas.update(model.potential)
    # DFs consistent with the final potential, used for sampling
    dfDisk, dfHalo = make_dfs()
    return model, dfDisk, dfHalo


def write_text(filename, posvel, mass, u=None):
    """Tab-separated file with header '#x y z vx vy vz mass [u]' (read by write_mirror_particles.py)."""
    cols, hdr = [posvel, mass], 'x\ty\tz\tvx\tvy\tvz\tmass'
    if u is not None:
        cols.append(u); hdr += '\tu'
    np.savetxt(filename, np.column_stack(cols), fmt='%.8g', delimiter='\t', header=hdr, comments='#')


def write_diagnostics(prefix, model, dfDisk, cfg, Rdisk):
    """Rotation curve and disc kinematic profiles (text + PNG)."""
    pot = model.potential
    R = np.linspace(0.05, 30, 120)
    xyz = np.column_stack((R, R * 0, R * 0))
    vcirc_tot = np.sqrt(-R * pot.force(xyz)[:, 0])
    vcirc_disk = np.sqrt(np.maximum(0, -R * model.components[0].getPotential().force(xyz)[:, 0])) \
        if hasattr(model.components[0], 'getPotential') else None

    # disc moments in the midplane
    gm = agama.GalaxyModel(potential=pot, df=dfDisk, af=model.af)
    Rm = np.linspace(0.25, 4 * Rdisk * 2.5, 30)
    pts = np.column_stack((Rm, Rm * 0, Rm * 0))
    dens, meanvel, vel2 = gm.moments(pts, dens=True, vel=True, vel2=True)
    vphi = meanvel[:, 1]
    sigR = np.sqrt(vel2[:, 0])
    sigz = np.sqrt(vel2[:, 2])
    sigphi = np.sqrt(np.maximum(0, vel2[:, 1] - vphi ** 2))
    # surface density: integrate the self-consistent disc density over z (Gauss-Legendre in z = zmax*t^2)
    densDisk = model.components[0].density
    t, w = np.polynomial.legendre.leggauss(40)
    t = 0.5 * (t + 1); w = 0.5 * w
    zmax = 10.0
    zz = zmax * t ** 2; wz = w * 2 * zmax * t
    grid = np.column_stack((np.repeat(Rm, len(zz)), np.zeros(len(Rm) * len(zz)), np.tile(zz, len(Rm))))
    Sigma = 2 * (densDisk.density(grid).reshape(len(Rm), len(zz)) * wz).sum(axis=1)

    np.savetxt(prefix + '_rotcurve.txt', np.column_stack((R, vcirc_tot)),
               fmt='%.6g', header='R[kpc]\tvcirc_total[km/s]')
    np.savetxt(prefix + '_diskprofile.txt',
               np.column_stack((Rm, Sigma, dens, vphi, sigR, sigphi, sigz)), fmt='%.6g',
               header='R[kpc]\tSigma[Msun/kpc2]\trho_midplane[Msun/kpc3]\tvphi\tsigmaR\tsigmaPhi\tsigmaZ [km/s]')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(R, vcirc_tot, 'k', label='v$_{circ}$ total')
    ax[0].plot(Rm, vphi, 'r', label=r'$\langle v_\phi \rangle$ disc')
    ax[0].set_xlabel('R [kpc]'); ax[0].set_ylabel('km/s'); ax[0].legend()
    ax[1].plot(Rm, sigR, label=r'$\sigma_R$')
    ax[1].plot(Rm, sigphi, label=r'$\sigma_\phi$')
    ax[1].plot(Rm, sigz, label=r'$\sigma_z$')
    ax[1].set_xlabel('R [kpc]'); ax[1].set_ylabel('km/s'); ax[1].legend()
    if np.all(np.isfinite(Sigma)):
        Md = float(cfg['dfDisk'].get('mass', model.components[0].density.totalMass()))
        ax[2].semilogy(Rm, Sigma, 'b', label='model')
        ax[2].semilogy(Rm, Md / (2 * np.pi * Rdisk ** 2) * np.exp(-Rm / Rdisk), 'k--',
                       label='exp. target')
        ax[2].set_xlabel('R [kpc]'); ax[2].set_ylabel(r'$\Sigma$ [M$_\odot$/kpc$^2$]'); ax[2].legend()
    fig.tight_layout()
    fig.savefig(prefix + '_diagnostics.png', dpi=120)
    plt.close(fig)


def even(n):
    n = int(n)
    return n + (n % 2)  # mirroring needs an even number of particles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ini', nargs='+', help='AGAMA .ini file(s) produced by make_AGAMA_ini.py')
    ap.add_argument('--ndisk', type=float, default=1e6, help='number of disc particles (default 1e6)')
    ap.add_argument('--nhalo', type=float, default=1e7, help='number of halo particles (default 1e7)')
    ap.add_argument('--niter', type=int, default=5, help='self-consistent iterations (default 5)')
    ap.add_argument('--outdir', default='.', help='output directory')
    ap.add_argument('--tag', default='', help='suffix added to output file names')
    ap.add_argument('--seed', type=int, default=None, help='random seed for sampling')
    ap.add_argument('--hdf5', action='store_true',
                    help='also write a mirrored Gadget HDF5 file (write_mirror_particles.mirror_agama)')
    ap.add_argument('--mass-unit', type=float, default=1.0,
                    help='mass unit [Msun] used in the HDF5 file; must match Gadget UnitMass_in_g '
                         '(default 1 = Msun, matching gadget4 param_*.txt UnitMass_in_g = 1.988e33)')
    ap.add_argument('--halo-cutoff', action='store_true',
                    help='apply outerCutoffRadius to the halo (truncated Hernquist) instead of ignoring it')
    ap.add_argument('--gas-frac', type=float, default=0.0,
                    help='gas fraction Mgas/(Mstars+Mgas); 0 = no gas (default). Stellar disc mass is unchanged')
    ap.add_argument('--gas-rd', type=float, default=None, help='gas scale length [kpc] (default: stellar Rdisk)')
    ap.add_argument('--gas-cs', type=float, default=10.0, help='gas isothermal sound speed [km/s] (default 10)')
    ap.add_argument('--gas-rtrunc', type=float, default=None, help='gas truncation radius [kpc] (default 12 Rd_gas)')
    ap.add_argument('--ngas', type=float, default=None, help='number of gas particles (default: same as --ndisk)')
    ap.add_argument('--gas-eos', choices=['isothermal', 'adiabatic'], default='isothermal',
                    help='sets u: isothermal -> u = cs^2 (Gadget-4 ISOTHERM_EQS); adiabatic -> u = cs^2/(gamma-1)')
    ap.add_argument('--gas-gamma', type=float, default=5. / 3, help='adiabatic index for --gas-eos adiabatic')
    ap.add_argument('--sub-frac', type=float, default=0.0,
                    help='fraction of the host halo mass in subhaloes; 0 = no substructure (default)')
    ap.add_argument('--sub-mmin', type=float, default=None, help='min subhalo mass [Msun] (default 100 particles)')
    ap.add_argument('--sub-mmax', type=float, default=None, help='max subhalo mass [Msun] (default 0.01 M_halo)')
    ap.add_argument('--sub-slope', type=float, default=-1.9, help='mass function slope dN/dm ~ m^slope')
    ap.add_argument('--sub-rb', type=float, default=None,
                    help='anti-bias radius [kpc]: local subhalo fraction ~ r/(r+r_b) (default 0.5 r200)')
    ap.add_argument('--sub-rcut', type=float, default=None, help='outer cutoff of subhalo distribution [kpc] (default r200)')
    ap.add_argument('--sub-c0', type=float, default=10.0, help='subhalo concentration c200 at 1e12 Msun (c ~ m^-0.1)')
    ap.add_argument('--softening', type=float, default=0.1, help='DM softening [kpc], only used for warnings')
    ap.add_argument('--quiet-start', action='store_true',
                    help='sample all components by action-space stratification (quietstart.py) + mirroring')
    ap.add_argument('--qs-oversample', type=float, default=2.0,
                    help='quiet start: request oversample*N/2 pool points (AGAMA returns ~5x more); >= 1 '
                         'guarantees no duplicates (default 2)')
    ap.add_argument('--no-diagnostics', action='store_true', help='skip rotation curve/profile output')
    args = ap.parse_args()

    if args.seed is not None and hasattr(agama, 'setRandomSeed'):
        agama.setRandomSeed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    for ini in args.ini:
        name = os.path.splitext(os.path.basename(ini))[0].replace('AGAMA_', '')
        tag = name + ('_' + args.tag if args.tag else '')
        print('=' * 70 + '\nModel %s  (from %s)' % (tag, ini))
        cfg = read_ini(ini, args.halo_cutoff)
        gas = None
        if args.gas_frac > 0:
            from gas_disc import GasDisc
            Mstar = float(cfg['dfDisk']['mass'])
            Mgas = Mstar * args.gas_frac / (1 - args.gas_frac)
            Rdg = args.gas_rd or float(cfg['dfDisk'].get('Rdisk', 2.0))
            gas = GasDisc(Mgas, Rdg, args.gas_cs, Rtrunc=args.gas_rtrunc)
            print('  gas disc: M=%.4g Msun (f_gas=%.3g), Rd=%.3g kpc, cs=%.3g km/s, Rtrunc=%.3g kpc' %
                  (Mgas, args.gas_frac, Rdg, args.gas_cs, gas.Rtrunc))
        sub = None
        if args.sub_frac > 0:
            from substructure import SubhaloPopulation
            densHost = agama.Density(**cfg['potHalo'])
            Mhost = densHost.totalMass()
            sub = SubhaloPopulation(densHost, Mhost, f_sub=args.sub_frac, m_min=args.sub_mmin,
                                    m_max=args.sub_mmax, slope=args.sub_slope, r_b=args.sub_rb,
                                    r_cut=args.sub_rcut, c0=args.sub_c0, particle_mass=Mhost / args.nhalo,
                                    rng=np.random.default_rng(args.seed), even_npart=args.quiet_start)
            print('  substructure: ' + sub.summary())
        t0 = time.time()
        model, dfDisk, dfHalo = build_model(cfg, niter=args.niter, gas=gas, sub=sub)
        print('Model built in %.1f s' % (time.time() - t0))

        prefix = os.path.join(args.outdir, 'model_' + tag)
        if not args.no_diagnostics:
            Rdisk = float(cfg['dfDisk'].get('Rdisk', 2.0))
            write_diagnostics(prefix, model, dfDisk, cfg, Rdisk)
        model.potential.export(prefix + '_potential.ini')

        nhalo = even(args.nhalo - (sub.npart.sum() if sub is not None else 0))  # smooth-halo particles
        print('Sampling %d disc and %d %shalo particles...' % (even(args.ndisk), nhalo, 'smooth ' if sub else ''))
        t0 = time.time()
        gmD = agama.GalaxyModel(potential=model.potential, df=dfDisk, af=model.af)
        gmH = agama.GalaxyModel(potential=model.potential, df=dfHalo, af=model.af)
        rng = np.random.default_rng(args.seed)
        if args.quiet_start:
            from quietstart import quiet_sample
            posvelD, massD, _ = quiet_sample(lambda n: gmD.sample(n, method=1) + (None,), model.af,
                                             even(args.ndisk) // 2, rng, oversample=args.qs_oversample, label='disc')
            posvelH, massH, _ = quiet_sample(lambda n: gmH.sample(n, method=1) + (None,), model.af,
                                             nhalo // 2, rng, oversample=args.qs_oversample, label='halo')
        else:
            posvelD, massD = gmD.sample(even(args.ndisk))
            posvelH, massH = gmH.sample(nhalo)
        print('  sampled in %.1f s; disc mass %.4g, halo mass %.4g Msun' %
              (time.time() - t0, massD.sum(), massH.sum()))

        fstars = os.path.join(args.outdir, 'model_stars_final_%s.txt' % tag)
        fdm = os.path.join(args.outdir, 'model_dm_final_%s.txt' % tag)
        write_text(fstars, posvelD, massD)
        write_text(fdm, posvelH, massH)
        print('  wrote %s\n  wrote %s' % (fstars, fdm))

        fgas = None
        if gas is not None:
            ngas = even(args.ngas if args.ngas else args.ndisk)
            gamma = None if args.gas_eos == 'isothermal' else args.gas_gamma
            if args.quiet_start:
                posvelG, massG, uG = quiet_sample(lambda n: gas.sample_pool(n, gamma=gamma), model.af,
                                                  ngas // 2, rng, oversample=args.qs_oversample, label='gas')
                fneg = np.mean((posvelG[:, 3] ** 2 + posvelG[:, 4] ** 2) == 0)
            else:
                posvelG, massG, uG, fneg = gas.sample(ngas, gas.density, gamma=gamma)
            fgas = os.path.join(args.outdir, 'model_gas_final_%s.txt' % tag)
            write_text(fgas, posvelG, massG, uG)
            print('  sampled %d gas particles (mass %.4g Msun, u=%.4g (km/s)^2, %s EOS); '
                  'v_phi^2<0 (set to 0) for %.2g%% of them' %
                  (ngas, massG.sum(), uG[0], args.gas_eos, 100 * fneg))
            print('  wrote %s' % fgas)

        fsub = fcat = None
        if sub is not None:
            t0 = time.time()
            S = sub.sample(model.potential, softening=args.softening, quiet=args.quiet_start, rng=rng,
                           oversample=args.qs_oversample)
            fsub = os.path.join(args.outdir, 'model_sub_final_%s.txt' % tag)
            np.savetxt(fsub, np.column_stack((S['pos'], S['vel'], S['mass'], S['subid'])),
                       fmt=['%.8g'] * 7 + ['%d'], delimiter='\t',
                       header='x\ty\tz\tvx\tvy\tvz\tmass\tsubid', comments='#')
            fcat = os.path.join(args.outdir, 'model_subhalos_%s.txt' % tag)
            cat = S['catalogue']
            print('  sampled %d subhaloes (%d particles) in %.1f s; centres at r = %.1f-%.1f kpc '
                  '(median %.1f), tidal radii %.2f-%.2f kpc' %
                  (len(cat), len(S['mass']), time.time() - t0, cat[:, 9].min(), cat[:, 9].max(),
                   np.median(cat[:, 9]), cat[:, 11].min(), cat[:, 11].max()))
            for w in S['warnings']:
                print('  WARNING: ' + w)
            print('  wrote %s' % fsub)

        if args.hdf5 and (sub is not None or args.quiet_start):
            from write_mirror_particles import mirror_agama_all
            fh5 = os.path.join(args.outdir, 'model_final_%s.hdf5' % tag)
            # quiet start: particles are already mirrored, so the writer must not mirror again
            ids = mirror_agama_all(fstars, fdm, fh5, gas_file_in=fgas, sub=S if sub is not None else None,
                                   mass_unit=args.mass_unit, mirror=not args.quiet_start)
            print('  wrote %s (masses in units of %.3g Msun)' % (fh5, args.mass_unit))
            if sub is not None:
                # catalogue with the ParticleID range of each subhalo (PartType3, contiguous per subhalo)
                first = ids[3] + np.r_[0, np.cumsum(cat[:, 2])[:-1]].astype(np.int64)
                np.savetxt(fcat, np.column_stack((cat, first, first + cat[:, 2] - 1)),
                           fmt=['%d', '%.6g', '%d'] + ['%.6g'] * 10 + ['%d', '%d'], delimiter='\t',
                           header=sub.CAT_HEADER + '\tpid_first\tpid_last', comments='#')
                print('  wrote %s' % fcat)
        elif sub is not None:
            np.savetxt(fcat, S['catalogue'], fmt=['%d', '%.6g', '%d'] + ['%.6g'] * 10, delimiter='\t',
                       header=sub.CAT_HEADER, comments='#')
            print('  wrote %s' % fcat)
        elif args.hdf5:
            fh5 = os.path.join(args.outdir, 'model_final_%s.hdf5' % tag)
            if fgas is None:
                from write_mirror_particles import mirror_agama
                mirror_agama(fstars, fdm, fh5, mass_unit=args.mass_unit)
            else:
                from write_mirror_particles import mirror_agama_wgas
                mirror_agama_wgas(fstars, fdm, fgas, fh5, mass_unit=args.mass_unit)
            print('  wrote %s (masses in units of %.3g Msun)' % (fh5, args.mass_unit))


if __name__ == '__main__':
    main()
