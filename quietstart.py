"""
Quiet-start particle sampling by stratification in action space.

The same prescription is used for every component (stellar disc, halo, gas, each subhalo):

  1. Draw a weighted pool of candidates from the component's distribution with AGAMA's
     quasi-random sampler (method=1: low-discrepancy points with unequal weights; their
     weighted sums are far less noisy than independent draws). Requesting oversample*n points
     returns ~5x more, with every weight <= W/(oversample*n).
  2. Compute the three actions (J_R, J_z, J_phi) of every candidate in the relevant potential.
  3. Replace each action by its weighted rank within the pool, scaled to [0, 1)  (a copula
     transform: removes units/ranges so the three actions are treated on an equal footing).
  4. Order the pool along a 3D Hilbert curve through that unit cube (neighbours along the
     curve are neighbours in action space).
  5. Systematic resampling along the curve: cut the cumulative weight into n equal blocks and
     take the candidate at the same random offset within every block. The selected, equal-mass
     particles sample the action distribution evenly in all three dimensions at once; since
     weights <= W/n, no candidate can be selected twice.
  6. (Done by the caller.) Point-mirror the selection, (x, v) -> (-x, -v), which leaves all
     three actions unchanged and only shifts the azimuthal and vertical angles by pi, so the
     stratification is preserved while odd angular modes and the centre-of-mass offset cancel.

Rationale: in an axisymmetric equilibrium the DF depends only on the actions (Jeans theorem);
the angles carry no equilibrium information. Stratifying in actions fixes the equilibrium
content of the realisation (e.g. M(<r), Sigma(R), vertical structure), leaving randomness only
in the angles. For gas (pressure supported, v_R = v_z = 0 initially) the actions computed in the
gravitational potential are not dynamical invariants, but they are monotonic labels of radius
and height, so the same code stratifies the gas in (R, |z|).

Candidates whose actions cannot be computed (e.g. beyond the potential grid) are kept and
assigned the largest ranks, so the outer tail of the distribution is not removed.
"""
import numpy as np


def hilbert_index_3d(coords, bits):
    """Hilbert-curve index of integer coordinates (N x 3, each in [0, 2^bits)).
    Skilling (2004, AIP Conf. Proc. 707, 381) 'AxestoTranspose', vectorised; 3*bits <= 63."""
    X = [coords[:, i].astype(np.uint64).copy() for i in range(3)]
    n = 3
    M = np.uint64(1) << np.uint64(bits - 1)
    # inverse undo excess work
    Q = M
    while Q > 1:
        P = Q - np.uint64(1)
        for i in range(n):
            has = (X[i] & Q) != 0
            # if bit set: invert low bits of X[0]; else: exchange low bits of X[0] and X[i]
            t = (X[0] ^ X[i]) & P
            X0_new = np.where(has, X[0] ^ P, X[0] ^ t)
            Xi_new = np.where(has, X[i], X[i] ^ t)
            X[0] = X0_new
            if i != 0:
                X[i] = Xi_new
        Q >>= np.uint64(1)
    # Gray encode
    for i in range(1, n):
        X[i] ^= X[i - 1]
    t = np.zeros_like(X[0])
    Q = M
    while Q > 1:
        t = np.where((X[n - 1] & Q) != 0, t ^ (Q - np.uint64(1)), t)
        Q >>= np.uint64(1)
    for i in range(n):
        X[i] ^= t
    # interleave the transposed bits (most significant first) into a single index
    h = np.zeros_like(X[0])
    for b in range(bits - 1, -1, -1):
        for i in range(n):
            h = (h << np.uint64(1)) | ((X[i] >> np.uint64(b)) & np.uint64(1))
    return h


def rank_uniform(a, w=None):
    """(Weighted) ranks of a 1D array scaled to [0, 1): mid-point of each element's share of
    the cumulative weight. Non-finite values get the largest ranks."""
    w = np.ones(len(a)) if w is None else w
    a = np.where(np.isfinite(a), a, np.inf)
    o = np.argsort(a, kind='stable')
    c = np.cumsum(w[o])
    u = np.empty(len(a))
    u[o] = (c - 0.5 * w[o]) / c[-1]
    return u


def stratified_select(actions, n, rng, weights=None, bits=21):
    """Indices of n candidates chosen by systematic resampling along the Hilbert curve through
    the (weighted-)rank-transformed action space. actions: (Npool x 3); weights: (Npool,)."""
    npool = len(actions)
    w = np.ones(npool) if weights is None else np.asarray(weights, dtype=float)
    U = np.column_stack([rank_uniform(actions[:, k], w) for k in range(3)])
    cells = np.minimum((U * (1 << bits)).astype(np.int64), (1 << bits) - 1)
    order = np.argsort(hilbert_index_3d(cells, bits), kind='stable')
    cw = np.cumsum(w[order])
    targets = (np.arange(n) + rng.random()) * cw[-1] / n  # one common offset -> systematic
    return order[np.searchsorted(cw, targets, side='right').clip(0, npool - 1)]


def cdf_deviation(sample, reference, ref_weights=None):
    """Max |CDF_sample - CDF_reference| for each column (a KS-like distance); the reference
    may be weighted, the sample is equal-weight."""
    w = np.ones(len(reference)) if ref_weights is None else ref_weights
    out = []
    for k in range(sample.shape[1]):
        s = np.sort(np.where(np.isfinite(sample[:, k]), sample[:, k], np.inf))
        ra = np.where(np.isfinite(reference[:, k]), reference[:, k], np.inf)
        o = np.argsort(ra); ref = ra[o]; cwr = np.cumsum(w[o]) / w.sum()
        g = np.linspace(0, len(ref) - 1, min(len(ref), 20000)).astype(int)
        cs = np.searchsorted(s, ref[g], side='right') / len(s)
        out.append(np.max(np.abs(cs - cwr[g])))
    return np.array(out)


def radial_mass_deviation(xv_sel, xv_pool, w_pool, quantiles=(0.01, 0.03, 0.1, 0.3, 0.5, 0.9)):
    """Fractional error of the selection's enclosed mass M(<r) (spherical r) relative to the
    weighted pool, at the pool's mass quantiles."""
    rp = np.linalg.norm(xv_pool[:, :3], axis=1); rs = np.linalg.norm(xv_sel[:, :3], axis=1)
    o = np.argsort(rp); cw = np.cumsum(w_pool[o]) / w_pool.sum()
    radii = rp[o][np.searchsorted(cw, quantiles)]
    return radii, np.array([np.mean(rs < r) for r in radii]) / np.array(quantiles) - 1


def quiet_sample(sample_pool, actions_of, n_half, rng, oversample=2, bits=21, label='', verbose=True):
    """Quiet-start sample of 2*n_half equal-mass particles of one component.

    sample_pool(nreq) -> (xv[npool, 6], weight[npool], extra or None)
        weighted candidates from the component's distribution (AGAMA method=1 sampling, which
        returns ~5*nreq points with weights <= W/nreq); extra: per-particle arrays to carry
        along (e.g. gas internal energy) or None
    actions_of(xv) -> actions[npool, 3]  (J_R, J_z, J_phi)

    Returns xv[2 n_half, 6], mass[2 n_half], extra[2 n_half] (or None): the stratified half
    followed by its point-mirrored copy; total mass = total pool weight."""
    xv, w, extra = sample_pool(int(oversample * n_half))
    keep = w > 0
    xv, w = xv[keep], w[keep]
    extra = None if extra is None else extra[keep]
    J = actions_of(xv)
    idx = stratified_select(J, n_half, rng, weights=w, bits=bits)
    if verbose:
        rnd = rng.choice(len(w), n_half, replace=False, p=w / w.sum())  # reference: weighted random
        nbad = np.sum(~np.all(np.isfinite(J), axis=1))
        radii, dq = radial_mass_deviation(xv[idx], xv, w)
        _, dr = radial_mass_deviation(xv[rnd], xv, w)
        print('  quiet start %-9s pool %d -> %d (+ mirror), duplicates %d; max|dCDF| of (J_R, J_z, J_phi): '
              'stratified %s, random %s%s' %
              (label, len(w), n_half, n_half - len(np.unique(idx)),
               np.array2string(cdf_deviation(J[idx], J, w), precision=5),
               np.array2string(cdf_deviation(J[rnd], J, w), precision=5),
               '; %d candidates without finite actions kept at the top ranks' % nbad if nbad else ''))
        print('              M(<r) error at r = %s kpc: stratified %s, random %s' %
              (np.array2string(radii, precision=3), np.array2string(dq, precision=4),
               np.array2string(dr, precision=4)))
    sel = xv[idx]
    xv2 = np.concatenate((sel, -sel))
    mass = np.full(2 * n_half, w.sum() / (2 * n_half))
    ex2 = None if extra is None else np.concatenate((extra[idx], extra[idx]))
    return xv2, mass, ex2
