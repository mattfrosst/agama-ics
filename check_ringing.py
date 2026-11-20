"""
Look for "ringing" (coherent oscillations signalling out-of-equilibrium initial conditions) in
the stellar disc (PartType2) and gas disc (PartType0) of a high-cadence Gadget-4 run.

In annuli around several radii, for each snapshot:
  thickness  median |z|                        -> vertical breathing, expected period ~ P_z/2
  <z v_z>/(rms z * sigma_z)                    -> breathing phase (0 in equilibrium, oscillates sign)
  <v_R>                                        -> radial ringing, expected period ~ P_kappa
  sigma_R, sigma_z                             -> dispersion breathing (~P_kappa/2, P_z/2)
  Sigma(t)/Sigma(0)                            -> radial mass redistribution
Everything is measured relative to the stellar disc's centre of mass and mean velocity.

If --potential (an AGAMA potential .ini exported by run_AGAMA_model.py) is given, the vertical
(P_z = 2pi/nu), epicyclic (P_kappa = 2pi/kappa) periods at each radius are computed and compared
with the dominant period of each detrended time series (FFT; coarse if the run spans few periods).

Usage:
    python check_ringing.py /path/to/output [--potential model_<tag>_potential.ini] [--radii 1 2 4 8]
                            [--compare /path/to/other/output --labels this other]
    --compare overlays a second run (its ringing.npz, written by a previous call) and prints
    the ringing amplitudes of both: rms <v_R>, rms (Sigma/Sigma0 - 1), rms <z v_z>, and the
    rms of the detrended thickness.
"""
import os
import glob
import argparse
import numpy as np
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def annulus_stats(pos, vel, mass, R0, dR):
    R = np.hypot(pos[:, 0], pos[:, 1])
    s = np.abs(R - R0) < dR
    p, v = pos[s], vel[s]
    Rs = R[s]
    vR = (p[:, 0] * v[:, 0] + p[:, 1] * v[:, 1]) / Rs
    z, vz = p[:, 2], v[:, 2]
    rmsz, sz = np.sqrt(np.mean(z ** 2)), np.std(vz)
    return dict(n=s.sum(), Sigma=mass[s].sum() / (np.pi * ((R0 + dR) ** 2 - (R0 - dR) ** 2)),
                medz=np.median(np.abs(z)), zvz=np.mean(z * vz) / max(rmsz * sz, 1e-10),
                vR=np.mean(vR), sigR=np.std(vR), sigz=sz)


def dominant_period(t, y):
    """Period of the largest FFT peak of the linearly detrended series (None if too short)."""
    if len(t) < 8:
        return None
    y = y - np.polyval(np.polyfit(t, y, 1), t)
    dt = np.median(np.diff(t))
    ti = np.arange(t[0], t[-1], dt); yi = np.interp(ti, t, y) * np.hanning(len(ti))
    P = np.abs(np.fft.rfft(yi)); f = np.fft.rfftfreq(len(ti), dt)
    k = 1 + np.argmax(P[1:])
    return 1 / f[k] if f[k] > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('outdir')
    ap.add_argument('--potential', default=None)
    ap.add_argument('--radii', type=float, nargs='+', default=[1, 2, 4, 8])
    ap.add_argument('--compare', default=None, help='output dir of another run with ringing.npz')
    ap.add_argument('--labels', nargs=2, default=['this run', 'compare'])
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.outdir, 'snapshot_*.hdf5')))
    comps = {2: 'stars', 0: 'gas'}
    radii = args.radii
    dRs = [max(0.25, 0.125 * R) for R in radii]
    keys = ['medz', 'zvz', 'vR', 'sigR', 'sigz', 'Sigma']
    data = {c: {k: np.zeros((len(files), len(radii))) for k in keys} for c in comps}
    T = np.zeros(len(files))
    for i, fn in enumerate(files):
        with h5py.File(fn, 'r') as f:
            T[i] = f['Header'].attrs['Time']
            ms = f['PartType2/Masses'][:]
            xc = np.average(f['PartType2/Coordinates'][:], axis=0, weights=ms)
            vc = np.average(f['PartType2/Velocities'][:], axis=0, weights=ms)
            for c in comps:
                g = 'PartType%d' % c
                if g not in f:
                    continue
                pos = f[g]['Coordinates'][:] - xc; vel = f[g]['Velocities'][:] - vc; m = f[g]['Masses'][:]
                for j, (R0, dR) in enumerate(zip(radii, dRs)):
                    st = annulus_stats(pos, vel, m, R0, dR)
                    for k in keys:
                        data[c][k][i, j] = st[k]
    present = [c for c in comps if np.any(data[c]['Sigma'])]

    periods = None
    if args.potential:
        import agama
        agama.setUnits(mass=1, length=1, velocity=1)
        pot = agama.Potential(file=args.potential)
        periods = []
        for R in radii:
            _, d = pot.forceDeriv([R, 0, 0])  # (deprecated alias in AGAMA 1.0, still works)
            Om2 = -pot.force([R, 0, 0])[0] / R
            periods.append((2 * np.pi / np.sqrt(-d[2]), 2 * np.pi / np.sqrt(-d[0] + 3 * Om2)))

    # ---- text summary ----
    print('Snapshots: %d, t = %.4f - %.4f, dt = %.4f' % (len(T), T[0], T[-1], np.median(np.diff(T))))
    for c in present:
        print('\n%s' % comps[c].upper())
        print('  R    medz(0)->medz(end)  [min,max]/medz(0)   max|<zv_z>|   <v_R> rms [km/s]   Sigma(end)/Sigma(0)'
              '   P_dom(medz)  P_dom(v_R)' + ('   P_z/2   P_z   P_kap' if periods else ''))
        for j, R in enumerate(radii):
            d = {k: data[c][k][:, j] for k in keys}
            line = ('%4.1f   %.4f -> %.4f      [%.2f, %.2f]         %.3f          %5.2f             %.3f' %
                    (R, d['medz'][0], d['medz'][-1], d['medz'].min() / d['medz'][0], d['medz'].max() / d['medz'][0],
                     np.abs(d['zvz']).max(), np.sqrt(np.mean(d['vR'] ** 2)), d['Sigma'][-1] / d['Sigma'][0]))
            pm, pv = dominant_period(T, d['medz']), dominant_period(T, d['vR'])
            line += '          %s       %s' % ('%.4f' % pm if pm else '  -   ', '%.4f' % pv if pv else '  -   ')
            if periods:
                line += '    %.4f  %.4f  %.4f' % (periods[j][0] / 2, periods[j][0], periods[j][1])
            print(line)

    # ---- figure: rows = quantities, columns = components ----
    rows = [('medz', 'median |z| / t=0', True), ('zvz', r'$\langle z v_z\rangle / (z_{rms}\sigma_z)$', False),
            ('vR', r'$\langle v_R \rangle$ [km/s]', False), ('sigz', r'$\sigma_z$ / t=0', True),
            ('sigR', r'$\sigma_R$ / t=0', True), ('Sigma', r'$\Sigma / \Sigma(t=0)$', True)]
    fig, ax = plt.subplots(len(rows), len(present), figsize=(6.5 * len(present), 2.2 * len(rows)),
                           sharex=True, squeeze=False)
    cols = plt.get_cmap('plasma')(np.linspace(0.05, 0.8, len(radii)))
    for jc, c in enumerate(present):
        for ir, (k, lab, norm) in enumerate(rows):
            a = ax[ir, jc]
            for j, R in enumerate(radii):
                y = data[c][k][:, j]
                if norm and abs(y[0]) > 1e-3 * np.abs(y).max():
                    y = y / y[0]
                lbl = 'R=%g' % R
                if periods and ir == 0:
                    lbl += r' (P$_z$/2=%.3f)' % (periods[j][0] / 2)
                a.plot(T, y, '.-', color=cols[j], ms=3, lw=1, label=lbl)
            y0 = data[c][k][0]
            if norm and np.all(np.abs(y0) <= 1e-3 * np.abs(data[c][k]).max()):
                lab = lab.replace(' / t=0', ' [km/s]')  # starts at zero (e.g. cold gas): plot absolute values
            if not norm:
                a.axhline(0, color='0.6', lw=0.6)
            a.set_ylabel(lab, fontsize=8); a.tick_params(labelsize=7)
            if ir == 0:
                a.set_title('%s disc' % comps[c]); a.legend(fontsize=6.5, ncol=2)
        ax[-1, jc].set_xlabel('t [kpc/(km/s) = 0.978 Gyr]')
    fig.tight_layout()
    out = os.path.join(args.outdir, 'ringing.png')
    fig.savefig(out, dpi=120)
    np.savez(os.path.join(args.outdir, 'ringing.npz'), t=T, radii=radii,
             **{'%s_%s' % (comps[c], k): data[c][k] for c in present for k in keys})
    print('\nwrote %s and ringing.npz' % out)
    if args.compare:
        compare_runs(args.outdir, args.compare, args.labels)


def amplitudes(t, d):
    """Ringing amplitudes for one component/radius: rms <v_R>, rms(Sigma/Sigma0-1),
    rms <z v_z> (normalised), rms of linearly detrended thickness/thickness0."""
    th = d['medz'] / d['medz'][0]
    th = th - np.polyval(np.polyfit(t, th, 1), t)
    return (np.sqrt(np.mean(d['vR'] ** 2)), np.sqrt(np.mean((d['Sigma'] / d['Sigma'][0] - 1) ** 2)),
            np.sqrt(np.mean(d['zvz'] ** 2)), np.sqrt(np.mean(th ** 2)))


def compare_runs(dir_a, dir_b, labels):
    A = np.load(os.path.join(dir_a, 'ringing.npz')); B = np.load(os.path.join(dir_b, 'ringing.npz'))
    radii = A['radii']
    comps = [c for c in ('stars', 'gas') if '%s_vR' % c in A.files and '%s_vR' % c in B.files]
    print('\nRinging amplitudes (%s vs %s):' % tuple(labels))
    print('  comp    R    rms<v_R> [km/s]     rms(Sigma/Sigma0-1)     rms<z v_z>/(z sz)     rms detrended thickness')
    for c in comps:
        for j, R in enumerate(radii):
            out = []
            for D in (A, B):
                out.append(amplitudes(D['t'], {k: D['%s_%s' % (c, k)][:, j] for k in ('vR', 'Sigma', 'zvz', 'medz')}))
            print('  %-6s %4.1f   %6.2f vs %6.2f      %.4f vs %.4f        %.3f vs %.3f         %.4f vs %.4f' %
                  (c, R, out[0][0], out[1][0], out[0][1], out[1][1], out[0][2], out[1][2], out[0][3], out[1][3]))
    rows = [('vR', r'$\langle v_R \rangle$ [km/s]', False), ('Sigma', r'$\Sigma/\Sigma_0$', True),
            ('medz', 'median |z| / t=0', True)]
    fig, ax = plt.subplots(len(rows), len(comps), figsize=(6.5 * len(comps), 2.4 * len(rows)), sharex=True,
                           squeeze=False)
    cols = plt.get_cmap('plasma')(np.linspace(0.05, 0.8, len(radii)))
    for jc, c in enumerate(comps):
        for ir, (k, lab, norm) in enumerate(rows):
            a = ax[ir, jc]
            for j, R in enumerate(radii):
                for D, ls, lb in ((A, '-', labels[0]), (B, ':', labels[1])):
                    y = D['%s_%s' % (c, k)][:, j]
                    y = y / y[0] if norm else y
                    a.plot(D['t'], y, ls, color=cols[j], lw=1.2,
                           label='R=%g %s' % (R, lb) if ir == 0 else None)
            if not norm:
                a.axhline(0, color='0.6', lw=0.6)
            a.set_ylabel(lab, fontsize=8); a.tick_params(labelsize=7)
            if ir == 0:
                a.set_title('%s disc: solid = %s, dotted = %s' % (c, labels[0], labels[1]), fontsize=9)
                a.legend(fontsize=6, ncol=2)
        ax[-1, jc].set_xlabel('t [kpc/(km/s) = 0.978 Gyr]')
    fig.tight_layout()
    out = os.path.join(dir_a, 'ringing_compare.png')
    fig.savefig(out, dpi=120)
    print('wrote', out)


if __name__ == '__main__':
    main()
