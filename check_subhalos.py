"""
Track subhaloes (PartType3) through a Gadget-4 run, using the catalogue written by
run_AGAMA_model.py (model_subhalos_<tag>.txt, with pid_first/pid_last columns).

For every subhalo and snapshot:
  * centre: shrinking-sphere centre of its particles
  * galactocentric radius of the centre (relative to the stellar-disc centre of mass)
  * f(<r_t): fraction of its particles within its initial tidal radius r_t of the centre
             (a simple proxy for the bound fraction; ~0.9 for an intact truncated NFW)
Writes subhalo_tracks.txt and subhalo_tracks.png in the output directory.

Usage:
    python check_subhalos.py /path/to/output /path/to/model_subhalos_<tag>.txt
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def shrinking_centre(x, start=None, shrink=0.8, nmin=20):
    c = np.median(x, axis=0) if start is None else start
    r = np.sqrt(((x - c) ** 2).sum(1)).max()
    sel = x
    while len(sel) > nmin:
        d = np.sqrt(((sel - c) ** 2).sum(1))
        sel = sel[d < r]
        if len(sel) == 0:
            break
        c = sel.mean(0)
        r *= shrink
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('catalogue')
    args = ap.parse_args()

    cat = np.loadtxt(args.catalogue)
    cat = np.atleast_2d(cat)
    sid, msub, rt0 = cat[:, 0].astype(int), cat[:, 1], cat[:, 11]
    pfirst, plast = cat[:, 13].astype(np.int64), cat[:, 14].astype(np.int64)
    snaps = sorted(glob.glob(os.path.join(args.outdir, 'snapshot_*.hdf5')))

    T, R, F = [], [], []
    prev = [None] * len(sid)
    for s in snaps:
        with h5py.File(s, 'r') as f:
            t = f['Header'].attrs['Time']
            ids = f['PartType3/ParticleIDs'][:]
            x = f['PartType3/Coordinates'][:]
            dc = np.average(f['PartType2/Coordinates'][:], axis=0, weights=f['PartType2/Masses'][:])
        order = np.argsort(ids); ids = ids[order]; x = x[order]
        rr, ff = [], []
        for i in range(len(sid)):
            a, b = np.searchsorted(ids, [pfirst[i], plast[i] + 1])
            xi = x[a:b]
            c = shrinking_centre(xi, prev[i])
            prev[i] = c
            rr.append(np.linalg.norm(c - dc))
            ff.append(np.mean(np.sqrt(((xi - c) ** 2).sum(1)) < rt0[i]))
        T.append(t); R.append(rr); F.append(ff)
    T, R, F = np.array(T), np.array(R), np.array(F)

    np.savetxt(os.path.join(args.outdir, 'subhalo_tracks.txt'),
               np.column_stack((np.repeat(T, len(sid)), np.tile(sid, len(T)), R.ravel(), F.ravel())),
               fmt=['%.5g', '%d', '%.5g', '%.4f'], header='t\tsubid\tr_centre[kpc]\tf(<r_t0)')
    print('subhalo  mass[Msun]  N   r_t0[kpc]  r(t0)->r(tend) [kpc]   f(<r_t0): t0 -> tend')
    for i in range(len(sid)):
        print('%5d  %9.3g  %5d  %7.2f   %7.1f -> %7.1f        %.3f -> %.3f' %
              (sid[i], msub[i], cat[i, 2], rt0[i], R[0, i], R[-1, i], F[0, i], F[-1, i]))
    print('median f(<r_t0): t0 %.3f -> tend %.3f' % (np.median(F[0]), np.median(F[-1])))

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cmap = plt.get_cmap('viridis')
    lm = np.log10(msub); norm = plt.Normalize(lm.min(), lm.max())
    for i in range(len(sid)):
        ax[0].plot(T, R[:, i], color=cmap(norm(lm[i])))
        ax[1].plot(T, F[:, i] / F[0, i], color=cmap(norm(lm[i])))
    ax[0].set_ylabel('galactocentric radius [kpc]'); ax[1].set_ylabel(r'f($<r_{t,0}$) / f(t=0)')
    for a in ax:
        a.set_xlabel('t [kpc/(km/s) = 0.978 Gyr]')
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, ax=ax, label=r'log$_{10}$ m$_{sub}$ [M$_\odot$]')
    fig.savefig(os.path.join(args.outdir, 'subhalo_tracks.png'), dpi=110, bbox_inches='tight')
    print('wrote subhalo_tracks.txt, subhalo_tracks.png in', args.outdir)


if __name__ == '__main__':
    main()
