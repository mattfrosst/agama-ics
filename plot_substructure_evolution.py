"""
Face-on (x-y) and edge-on (x-z) projections at every snapshot of a Gadget-4 run with
substructure, as three figures:

  1) evolution_disc.png      stellar disc (PartType2), surface density
  2) evolution_halo.png      primary (smooth) halo (PartType1), surface density
  3) evolution_subhalos.png  subhalo particles (PartType3), one colour per subhalo
                             (by log mass if the catalogue is given), with the disc size marked

Every panel is centred on the stellar disc's centre of mass at that snapshot. Each snapshot
gets a face-on/edge-on pair; pairs are laid out three per row. Surface-density maps share one
colour scale per figure (face-on and edge-on separately) so the evolution can be compared.

Usage:
    python plot_substructure_evolution.py /path/to/output [--catalogue model_subhalos_<tag>.txt]
        [--disc-extent 12 --disc-zext 4 --halo-extent 100 --sub-extent 200 --pairs-per-row 3]
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize


def read_snap(fname):
    out = {}
    with h5py.File(fname, 'r') as f:
        out['t'] = f['Header'].attrs['Time']
        for t in (1, 2, 3):
            g = 'PartType%d' % t
            if g in f:
                out[t] = dict(pos=f[g]['Coordinates'][:], mass=f[g]['Masses'][:], ids=f[g]['ParticleIDs'][:])
    c = np.average(out[2]['pos'], axis=0, weights=out[2]['mass'])  # disc centre
    for t in (1, 2, 3):
        if t in out:
            out[t]['pos'] = out[t]['pos'] - c
    return out


def layout(nsnap, ppr):
    nrows = int(np.ceil(nsnap / ppr))
    fig, axes = plt.subplots(nrows, 2 * ppr, figsize=(2.7 * 2 * ppr, 2.7 * nrows), squeeze=False)
    pairs = []
    for k in range(nrows * ppr):
        r, c = divmod(2 * k, 2 * ppr)
        if k < nsnap:
            pairs.append((axes[r, c], axes[r, c + 1]))
        else:
            axes[r, c].set_visible(False); axes[r, c + 1].set_visible(False)
    return fig, pairs


def label(axf, axe, t, k, zstretch=1.0):
    axf.set_title('#%03d  t=%.3f  face-on' % (k, t), fontsize=8)
    axe.set_title('t=%.3f  edge-on%s' % (t, '' if zstretch == 1 else ' (z x%.0f)' % zstretch), fontsize=8)
    axf.set_xlabel('x [kpc]', fontsize=7); axf.set_ylabel('y [kpc]', fontsize=7)
    axe.set_xlabel('x [kpc]', fontsize=7); axe.set_ylabel('z [kpc]', fontsize=7)
    for a in (axf, axe):
        a.tick_params(labelsize=6)


def density_figure(snaps, ptype, L, Z, nbins, title, outfile, ppr, cmap):
    nbz = max(10, int(round(nbins * Z / L)))
    maps = []
    for k, d in enumerate(snaps):
        p, m = d[ptype]['pos'], d[ptype]['mass']
        hf = np.histogram2d(p[:, 0], p[:, 1], bins=nbins, range=[[-L, L], [-L, L]], weights=m)[0] / (2 * L / nbins) ** 2
        he = np.histogram2d(p[:, 0], p[:, 2], bins=[nbins, nbz], range=[[-L, L], [-Z, Z]], weights=m)[0] / \
            ((2 * L / nbins) * (2 * Z / nbz))
        maps.append((hf, he))
    fv = np.concatenate([hf[hf > 0] for hf, _ in maps]); ev = np.concatenate([he[he > 0] for _, he in maps])
    nf = LogNorm(np.percentile(fv, 1), fv.max()); ne = LogNorm(np.percentile(ev, 1), ev.max())
    fig, pairs = layout(len(snaps), ppr)
    for k, ((axf, axe), (hf, he), d) in enumerate(zip(pairs, maps, snaps)):
        imf = axf.imshow(hf.T, origin='lower', extent=[-L, L, -L, L], norm=nf, cmap=cmap, interpolation='nearest')
        ime = axe.imshow(he.T, origin='lower', extent=[-L, L, -Z, Z], norm=ne, cmap=cmap, interpolation='nearest',
                         aspect='auto')
        for a in (axf, axe):
            a.set_facecolor('black')
        label(axf, axe, d['t'], k, L / Z)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.9, 0.98])
    for x0, img, lab in ((0.915, imf, 'face-on'), (0.955, ime, 'edge-on')):
        cax = fig.add_axes([x0, 0.15, 0.01, 0.7])
        cb = fig.colorbar(img, cax=cax); cb.ax.tick_params(labelsize=7)
        cb.set_label(r'%s $\Sigma$ [M$_\odot$ kpc$^{-2}$]' % lab, fontsize=8)
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print('wrote', outfile)


def subhalo_figure(snaps, catalogue, L, disc_R, outfile, ppr):
    if catalogue is not None:
        cat = np.atleast_2d(np.loadtxt(catalogue))
        pfirst, plast, lm = cat[:, 13].astype(np.int64), cat[:, 14].astype(np.int64), np.log10(cat[:, 1])
        norm = Normalize(lm.min(), lm.max()); cmap = plt.get_cmap('viridis')
    fig, pairs = layout(len(snaps), ppr)
    for k, ((axf, axe), d) in enumerate(zip(pairs, snaps)):
        p, ids = d[3]['pos'], d[3]['ids']
        if catalogue is not None:
            sub = np.searchsorted(pfirst, ids, side='right') - 1  # subhalo index of each particle
            col = cmap(norm(lm[np.clip(sub, 0, len(lm) - 1)]))
        else:
            col = 'C0'
        for a, j in ((axf, 1), (axe, 2)):
            a.scatter(p[:, 0], p[:, j], s=0.3, c=col, lw=0, rasterized=True)
            a.add_patch(plt.Rectangle((-disc_R, -disc_R if j == 1 else -0.5), 2 * disc_R,
                                      2 * disc_R if j == 1 else 1.0, fill=False, ec='r', lw=0.6))
            a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect('equal')
        label(axf, axe, d['t'], k)
    fig.suptitle('Subhaloes (PartType3); red box = stellar disc (|x|,|y| < %.0f kpc)' % disc_R, fontsize=11)
    fig.tight_layout(rect=[0, 0, 0.92 if catalogue is not None else 1, 0.98])
    if catalogue is not None:
        cax = fig.add_axes([0.935, 0.15, 0.01, 0.7])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
        cb.set_label(r'log$_{10}$ m$_{sub}$ [M$_\odot$]', fontsize=8); cb.ax.tick_params(labelsize=7)
    fig.savefig(outfile, dpi=120)
    plt.close(fig)
    print('wrote', outfile)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--catalogue', default=None, help='model_subhalos_<tag>.txt (colours subhaloes by mass)')
    ap.add_argument('--disc-extent', type=float, default=12.0)
    ap.add_argument('--disc-zext', type=float, default=4.0)
    ap.add_argument('--halo-extent', type=float, default=100.0)
    ap.add_argument('--sub-extent', type=float, default=200.0)
    ap.add_argument('--bins', type=int, default=150)
    ap.add_argument('--pairs-per-row', type=int, default=3)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.outdir, 'snapshot_*.hdf5')))
    if not files:
        raise SystemExit('no snapshots in ' + args.outdir)
    snaps = [read_snap(f) for f in files]
    ppr = args.pairs_per_row
    density_figure(snaps, 2, args.disc_extent, args.disc_zext, args.bins, 'Stellar disc (PartType2)',
                   os.path.join(args.outdir, 'evolution_disc.png'), ppr, 'inferno')
    density_figure(snaps, 1, args.halo_extent, args.halo_extent, args.bins, 'Primary (smooth) halo (PartType1)',
                   os.path.join(args.outdir, 'evolution_halo.png'), ppr, 'magma')
    if 3 in snaps[0]:
        subhalo_figure(snaps, args.catalogue, args.sub_extent, args.disc_extent,
                       os.path.join(args.outdir, 'evolution_subhalos.png'), ppr)
    else:
        print('no PartType3 particles: skipping subhalo figure')


if __name__ == '__main__':
    main()
