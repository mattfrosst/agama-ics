"""
Stability checks for an isolated disc+halo Gadget-4 run (HDF5 snapshots).

Tracks, per snapshot:
  - halo Lagrangian radii (10/50/90% mass)
  - disc half-mass radius, median |z| (thickness), sigma_z and sigma_R within R<2Rd..
  - bar strength A2 (m=2 Fourier amplitude) within 2 kpc
  - total energy drift (from output/energy.txt, if present)
and makes face-on / edge-on images of the disc for the first and last snapshots.

Usage:
    python check_gadget_run.py /path/to/output [--rd 2.0]
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load(fname):
    with h5py.File(fname, 'r') as f:
        t = f['Header'].attrs['Time']
        out = {'t': t}
        for name, pt in (('halo', 'PartType1'), ('disc', 'PartType2')):
            if pt in f:
                out[name] = (f[pt]['Coordinates'][:], f[pt]['Velocities'][:], f[pt]['Masses'][:])
    return out


def lagr_radii(pos, mass, centre, fracs=(0.1, 0.5, 0.9)):
    r = np.linalg.norm(pos - centre, axis=1)
    i = np.argsort(r)
    cm = np.cumsum(mass[i]) / mass.sum()
    return [r[i][np.searchsorted(cm, f)] for f in fracs]


def disc_stats(pos, vel, mass, Rd):
    centre = np.average(pos, axis=0, weights=mass)  # disc centre of mass
    vcen = np.average(vel, axis=0, weights=mass)
    p = pos - centre; v = vel - vcen
    R = np.hypot(p[:, 0], p[:, 1]); phi = np.arctan2(p[:, 1], p[:, 0])
    vR = (p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1]) / np.maximum(R, 1e-10)
    sel = (R > 0.5 * Rd) & (R < 1.5 * Rd)  # annulus around R = Rd
    inner = R < 2.0
    A2 = np.abs(np.sum(mass[inner] * np.exp(2j * phi[inner]))) / mass[inner].sum()
    return dict(centre=centre, Rhalf=lagr_radii(p, mass, 0, (0.5,))[0],
                medz=np.median(np.abs(p[sel, 2])), sigz=np.std(v[sel, 2]), sigR=np.std(vR[sel]),
                A2=A2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--rd', type=float, default=2.0, help='disc scale length [kpc]')
    args = ap.parse_args()

    snaps = sorted(glob.glob(os.path.join(args.outdir, 'snapshot_*.hdf5')))
    if not snaps:
        raise SystemExit('no snapshots in %s' % args.outdir)
    rows = []
    for s in snaps:
        d = load(s)
        ds = disc_stats(*d['disc'], args.rd)
        hp, hv, hm = d['halo']
        lr = lagr_radii(hp, hm, np.average(hp, axis=0, weights=hm))
        rows.append([d['t'], *lr, ds['Rhalf'], ds['medz'], ds['sigR'], ds['sigz'], ds['A2'],
                     np.linalg.norm(ds['centre'])])
    rows = np.array(rows)
    hdr = 't  halo_r10 r50 r90  disc_Rhalf  medz(R~Rd)  sigR(R~Rd)  sigz(R~Rd)  A2(R<2)  |disc_centre|'
    np.savetxt(os.path.join(args.outdir, 'stability.txt'), rows, fmt='%.5g', header=hdr)
    print(hdr)
    for r in rows[:: max(1, len(rows) // 10)].tolist() + [rows[-1].tolist()]:
        print('  '.join('%8.4g' % x for x in r))

    # energy conservation
    efile = os.path.join(args.outdir, 'energy.txt')
    if os.path.exists(efile):
        e = np.loadtxt(efile)
        # Gadget-4 energy.txt: time, Etot(thermal), Epot, Ekin, ...
        Etot = e[:, 2] + e[:, 3]
        print('Energy drift: dE/|E0| = %.2e (max %.2e)' %
              ((Etot[-1] - Etot[0]) / abs(Etot[0]), np.max(np.abs(Etot - Etot[0])) / abs(Etot[0])))

    # figures
    fig, ax = plt.subplots(2, 3, figsize=(14, 8))
    t = rows[:, 0]
    for k, lab in zip((1, 2, 3), ('10%', '50%', '90%')):
        ax[0, 0].plot(t, rows[:, k] / rows[0, k], label='halo r' + lab)
    ax[0, 0].plot(t, rows[:, 4] / rows[0, 4], 'k', label='disc R$_{1/2}$')
    ax[0, 0].set_ylabel('r / r(t=0)'); ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, rows[:, 5], label='median |z|'); ax[0, 1].set_ylabel('kpc (R~Rd)'); ax[0, 1].legend()
    ax[0, 2].plot(t, rows[:, 6], label=r'$\sigma_R$'); ax[0, 2].plot(t, rows[:, 7], label=r'$\sigma_z$')
    ax[0, 2].set_ylabel('km/s (R~Rd)'); ax[0, 2].legend()
    ax[1, 0].plot(t, rows[:, 8]); ax[1, 0].set_ylabel('A$_2$ (R<2 kpc)')
    for a in ax[0].tolist() + [ax[1, 0]]:
        a.set_xlabel('t [kpc/(km/s) = 0.978 Gyr]')
    for j, s in zip((1, 2), (snaps[0], snaps[-1])):
        d = load(s); p = d['disc'][0] - np.average(d['disc'][0], axis=0, weights=d['disc'][2])
        ax[1, j].hist2d(p[:, 0], p[:, 1], bins=200, range=[[-12, 12], [-12, 12]], cmap='inferno',
                        norm=matplotlib.colors.LogNorm())
        ax[1, j].set_aspect('equal'); ax[1, j].set_title('disc face-on, t=%.2f' % d['t'])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'stability.png'), dpi=110)

    fig, ax = plt.subplots(2, 1, figsize=(10, 5))
    for a, s in zip(ax, (snaps[0], snaps[-1])):
        d = load(s); p = d['disc'][0] - np.average(d['disc'][0], axis=0, weights=d['disc'][2])
        a.hist2d(p[:, 0], p[:, 2], bins=[300, 75], range=[[-12, 12], [-3, 3]], cmap='inferno',
                 norm=matplotlib.colors.LogNorm())
        a.set_aspect('equal'); a.set_title('disc edge-on, t=%.2f' % d['t'])
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, 'edgeon.png'), dpi=110)
    print('wrote stability.txt, stability.png, edgeon.png in', args.outdir)


if __name__ == '__main__':
    main()
