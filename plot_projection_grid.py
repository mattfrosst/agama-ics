"""
Grid of face-on and edge-on disc projections for every snapshot of a Gadget-4 run.

Each snapshot gets a pair of adjacent panels (face-on x-y, then edge-on x-z), filled
row by row into an nrows x ncols grid (ncols must be even). Unused panels are hidden.
Face-on and edge-on panels each use their own logarithmic colour scale (Msun/kpc^2),
shared across all snapshots so the time evolution can be compared directly.

Usage:
    python plot_projection_grid.py /path/to/output [--nrows 5 --ncols 6 --extent 12 --zext 4] [--ptype 0]
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def load_disc(fname, ptype=2):
    with h5py.File(fname, 'r') as f:
        t = f['Header'].attrs['Time']
        pos = f['PartType%d/Coordinates' % ptype][:]
        mass = f['PartType%d/Masses' % ptype][:]
    pos = pos - np.average(pos, axis=0, weights=mass)
    return t, pos, mass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--nrows', type=int, default=5)
    ap.add_argument('--ncols', type=int, default=6)
    ap.add_argument('--extent', type=float, default=12.0, help='half-width in x, y [kpc]')
    ap.add_argument('--zext', type=float, default=4.0, help='half-height of edge-on panels [kpc]')
    ap.add_argument('--bins', type=int, default=150)
    ap.add_argument('--ptype', type=int, default=2, help='particle type: 2 = stellar disc (default), 0 = gas')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    if args.ncols % 2:
        raise SystemExit('ncols must be even (face-on + edge-on pairs)')

    snaps = sorted(glob.glob(os.path.join(args.outdir, 'snapshot_*.hdf5')))
    npairs = args.nrows * args.ncols // 2
    if len(snaps) > npairs:
        print('warning: %d snapshots but only %d slots; plotting the first %d' % (len(snaps), npairs, npairs))
        snaps = snaps[:npairs]

    L, Z, nb = args.extent, args.zext, args.bins
    nbz = max(10, int(round(nb * Z / L)))
    face_area = (2 * L / nb) ** 2
    edge_area = (2 * L / nb) * (2 * Z / nbz)

    maps = []
    for s in snaps:
        t, p, m = load_disc(s, args.ptype)
        hf = np.histogram2d(p[:, 0], p[:, 1], bins=nb, range=[[-L, L], [-L, L]], weights=m)[0] / face_area
        he = np.histogram2d(p[:, 0], p[:, 2], bins=[nb, nbz], range=[[-L, L], [-Z, Z]], weights=m)[0] / edge_area
        maps.append((t, os.path.basename(s), hf, he))

    fv = np.concatenate([hf[hf > 0] for _, _, hf, _ in maps])
    ev = np.concatenate([he[he > 0] for _, _, _, he in maps])
    normf = LogNorm(vmin=np.percentile(fv, 1), vmax=fv.max())
    norme = LogNorm(vmin=np.percentile(ev, 1), vmax=ev.max())

    fig, axes = plt.subplots(args.nrows, args.ncols, figsize=(2.6 * args.ncols, 2.6 * args.nrows))
    axes = np.atleast_2d(axes)
    for k in range(npairs):
        r, c = divmod(2 * k, args.ncols)
        axf, axe = axes[r, c], axes[r, c + 1]
        if k >= len(maps):
            axf.set_visible(False); axe.set_visible(False)
            continue
        t, name, hf, he = maps[k]
        imf = axf.imshow(hf.T, origin='lower', extent=[-L, L, -L, L], norm=normf, cmap='inferno',
                        interpolation='nearest')
        ime = axe.imshow(he.T, origin='lower', extent=[-L, L, -Z, Z], norm=norme, cmap='inferno',
                   interpolation='nearest', aspect='auto')
        axf.set_title('%s  t=%.3f (face-on)' % (name.replace('snapshot_', '#').replace('.hdf5', ''), t), fontsize=8)
        axe.set_title('t=%.3f (edge-on, z ×%.0f)' % (t, L / Z), fontsize=8)
        for a in (axf, axe):
            a.tick_params(labelsize=6)
            a.set_facecolor('black')
        axf.set_xlabel('x [kpc]', fontsize=7); axf.set_ylabel('y [kpc]', fontsize=7)
        axe.set_xlabel('x [kpc]', fontsize=7); axe.set_ylabel('z [kpc]', fontsize=7)

    fig.tight_layout(rect=[0, 0, 0.90, 1])
    for x0, img, lab in ((0.915, imf, 'face-on'), (0.955, ime, 'edge-on')):
        cax = fig.add_axes([x0, 0.15, 0.01, 0.7])
        cb = fig.colorbar(img, cax=cax)
        cb.ax.tick_params(labelsize=7)
        cb.set_label(r'%s $\Sigma$ [M$_\odot$ kpc$^{-2}$]' % lab, fontsize=8)
    out = args.out or os.path.join(args.outdir, 'projection_grid%s.png' % ('' if args.ptype == 2 else '_type%d' % args.ptype))
    fig.savefig(out, dpi=130)
    print('wrote', out, '(%d snapshots)' % len(maps))


if __name__ == '__main__':
    main()
