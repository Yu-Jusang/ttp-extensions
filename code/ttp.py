"""
Core library for multi-source and shrinking-radius distributional test-then-pool.

Notation follows Yang, Liu and Evans (2026, arXiv:2603.11867):
  current control  Q_c (m obs),  treatment Q_t (n obs),
  historical source j  Q_{h_j} (ell_j obs),  kernel mean embedding xi.
D(P,Q) = ||xi_P - xi_Q||_H is the MMD (NOT squared); all test statistics below
are squared MMDs D^2 computed with the biased V-statistic (empirical embeddings).

All resampling is expressed through weight vectors on a precomputed Gram matrix,
so a "row" b = 0..B of a permutation scheme is a control/treatment allocation of
the current pool and row 0 is the observed allocation.
"""
import numpy as np

# ----------------------------------------------------------------------------
# data generation
# ----------------------------------------------------------------------------

def sample(rng, spec, size):
    """spec: ('norm', mu, sd) or ('mix', [(w, mu, sd), ...])."""
    kind = spec[0]
    if kind == 'norm':
        return rng.normal(spec[1], spec[2], size)
    if kind == 'mix':
        comps = spec[1]
        w = np.array([c[0] for c in comps], float)
        z = rng.choice(len(comps), size=size, p=w / w.sum())
        mu = np.array([c[1] for c in comps])[z]
        sd = np.array([c[2] for c in comps])[z]
        return rng.normal(mu, sd)
    raise ValueError(kind)


def rbf(x, y, sigma):
    d = x[:, None] - y[None, :]
    return np.exp(-d * d / (2.0 * sigma * sigma))


def median_heuristic_sigma(x):
    """sigma^2 = median of squared pairwise distances (v != v')."""
    d = (x[:, None] - x[None, :]) ** 2
    iu = np.triu_indices(len(x), 1)
    return float(np.sqrt(np.median(d[iu])))


# population MMD between N(0,1) and N(mu,1) for the RBF kernel (closed form)
def mmd_normal_shift(mu, sigma=1.0):
    s2 = sigma * sigma
    a = np.sqrt(s2 / (s2 + 2.0))
    return float(np.sqrt(max(0.0, 2 * a * (1 - np.exp(-mu * mu / (2 * (s2 + 2.0)))))))


def mu_for_mmd(target, sigma=1.0):
    """inverse of mmd_normal_shift in mu >= 0."""
    s2 = sigma * sigma
    a = np.sqrt(s2 / (s2 + 2.0))
    v = 1 - target * target / (2 * a)
    if v <= 0:
        raise ValueError('target too large')
    return float(np.sqrt(-2 * (s2 + 2.0) * np.log(v)))


# ----------------------------------------------------------------------------
# problem container
# ----------------------------------------------------------------------------

class Problem:
    def __init__(self, xc, xt, xhs, sigma):
        self.m, self.n = len(xc), len(xt)
        self.N = self.m + self.n
        self.J = len(xhs)
        self.ell = np.array([len(h) for h in xhs], dtype=int)
        Z = np.concatenate([xc, xt])
        allx = np.concatenate([Z] + list(xhs))
        K = rbf(allx, allx, sigma)
        N = self.N
        self.KZZ = K[:N, :N]
        self.hslices = []
        s = N
        for l in self.ell:
            self.hslices.append(slice(s, s + l))
            s += l
        self.KHZ = [K[sl, :N] for sl in self.hslices]           # ell_j x N
        self.KHH = [K[sl, sl] for sl in self.hslices]
        J = self.J
        self.r = np.stack([k.mean(0) for k in self.KHZ], 1) if J else np.zeros((N, 0))
        HH = np.zeros((J, J))
        for i in range(J):
            for j in range(J):
                HH[i, j] = K[self.hslices[i], self.hslices[j]].mean()
        self.HH = HH
        # unbiased RKHS trace variance of each historical source (k(x,x)=1)
        self.vh = np.array([(l / (l - 1.0)) * (1.0 - HH[j, j]) for j, l in enumerate(self.ell)])


def allocations(rng, N, m, B):
    """L x N boolean control indicators; row 0 = observed (first m are control)."""
    L = B + 1
    mask = np.zeros((L, N), dtype=bool)
    mask[0, :m] = True
    keys = rng.random((B, N))
    idx = np.argpartition(keys, m - 1, axis=1)[:, :m]
    rows = np.repeat(np.arange(1, L), m)
    mask[rows, idx.ravel()] = True
    return mask


class RowStats:
    """Inner products of empirical embeddings for every allocation row of a pool P."""

    def __init__(self, prob, P, mask):
        self.prob = prob
        self.P = P
        self.mask = mask
        self.m = int(mask[0].sum())
        self.n = mask.shape[1] - self.m
        C = mask / self.m
        T = (~mask) / self.n
        KP = prob.KZZ[np.ix_(P, P)]
        CK = C @ KP
        self.cc = np.einsum('ij,ij->i', CK, C)
        self.ct = np.einsum('ij,ij->i', CK, T)
        self.tt = np.einsum('ij,ij->i', T @ KP, T)
        rP = prob.r[P]
        self.ch = C @ rP          # L x J  <xi_c, xi_hj>
        self.th = T @ rP
        self.HH = prob.HH
        # Gram matrix of empirical shift directions  G_ij = <xi_hi - xi_c, xi_hj - xi_c>
        self.G = (prob.HH[None, :, :] - self.ch[:, :, None] - self.ch[:, None, :]
                  + self.cc[:, None, None])
        self.vc = (self.m / (self.m - 1.0)) * (1.0 - self.cc)   # unbiased trace variance
        self.Tcur = self.cc - 2 * self.ct + self.tt

    def Tbor(self, W):
        """W: L x J (row-specific) or J (frozen) weights; returns D^2(Q_f(w), Q_t) per row."""
        W = np.broadcast_to(np.asarray(W, float), self.ch.shape)
        s = W.sum(1)
        quad = np.einsum('li,ij,lj->l', W, self.HH, W)
        return ((1 - s) ** 2 * self.cc + quad + self.tt
                + 2 * (1 - s) * np.einsum('lj,lj->l', W, self.ch)
                - 2 * (1 - s) * self.ct - 2 * np.einsum('lj,lj->l', W, self.th))

    def disp2(self, W):
        """D^2(Q_f(w), Q_c^m) per row."""
        W = np.broadcast_to(np.asarray(W, float), self.ch.shape)
        return np.einsum('li,lij,lj->l', W, self.G, W)


# ----------------------------------------------------------------------------
# p-values
# ----------------------------------------------------------------------------

def perm_pvalue(T):
    """p = (1/L) #{b: T_b >= T_0}, ties counted (conservative)."""
    tol = 1e-12 * max(1.0, abs(T[0]))
    return float(np.mean(T >= T[0] - tol))


def row_pvalues(T):
    """rank p-value of every row within the L rows (ties counted)."""
    L = len(T)
    order = np.sort(T)
    tol = 1e-12 * max(1.0, np.abs(T).max())
    # number of entries >= T_l - tol
    cnt = L - np.searchsorted(order, T - tol, side='left')
    return cnt / L


def minp_pvalue(*Ts):
    """Westfall-Young style minP calibration across the same permutation rows."""
    U = np.min(np.stack([row_pvalues(T) for T in Ts]), 0)
    return float(np.mean(U <= U[0] + 1e-15))


# ----------------------------------------------------------------------------
# weighting (study A-1)
# ----------------------------------------------------------------------------

def simplex_grid(J, step=0.05, smax=0.9):
    k = int(round(1 / step))
    kmax = int(round(smax / step))
    pts = [()]
    for _ in range(J):
        pts = [p + (a,) for p in pts for a in range(k + 1)]
    pts = [p for p in pts if sum(p) <= kmax]
    pts.sort(key=lambda p: (sum(p), p))     # zero vector first -> ties resolved to no borrowing
    return np.array(pts, float) * step


def select_weights(rs, grid, kind='unbiased', vh=None, ell=None):
    """Row-wise argmin of an empirical risk objective over the grid (which contains 0).

    unbiased:     R_U(w) = w'Gw + (1-2s) vc/m      (unbiased for the fixed-w embedding MSE)
    conservative: L(w)   = w'Gw + (1-s)^2 vc/m + sum_j w_j^2 vh_j/ell_j   (plug-in proxy)
    """
    quad = np.einsum('gi,lij,gj->lg', grid, rs.G, grid)
    s = grid.sum(1)
    vcm = (rs.vc / rs.m)[:, None]
    if kind == 'unbiased':
        obj = quad + (1 - 2 * s)[None, :] * vcm
    elif kind == 'conservative':
        obj = quad + ((1 - s) ** 2)[None, :] * vcm + (grid ** 2 @ (vh / ell))[None, :]
    else:
        raise ValueError(kind)
    g = np.argmin(obj, 1)          # first minimiser: grid sorted with 0 first
    return grid[g]


# ----------------------------------------------------------------------------
# equivalence fusion test (Liu & Gandy 2026, as used in Yang et al. 2026)
# ----------------------------------------------------------------------------

def boot_hist_dists(prob, Bf, rng):
    """D_{W~}(Q_hj^ell) for Bf multinomial weight vectors, per source: J x Bf."""
    out = np.zeros((prob.J, Bf))
    for j in range(prob.J):
        l = prob.ell[j]
        E = rng.multinomial(l, np.full(l, 1.0 / l), size=Bf) - 1.0
        out[j] = np.sqrt(np.maximum(0, np.einsum('bi,bi->b', E @ prob.KHH[j], E))) / l
    return out


def boot_ctrl_dists_rows(prob, P, mask, rows, Bf, rng):
    """D_W(Q_c^m) of each listed row, common multinomial weights across rows: len(rows) x Bf."""
    m = int(mask[0].sum())
    E = rng.multinomial(m, np.full(m, 1.0 / m), size=Bf) - 1.0
    out = np.zeros((len(rows), Bf))
    for k, l in enumerate(rows):
        idx = P[mask[l]]
        Kc = prob.KZZ[np.ix_(idx, idx)]
        out[k] = np.sqrt(np.maximum(0, np.einsum('bi,bi->b', E @ Kc, E))) / m
    return out


def boot_ctrl_dists_pooled(prob, P, m, Bf, rng):
    """Label-invariant variant: draw m points from the whole current pool P and
    measure D(Q*_m, Q_P). Same value in every row."""
    Np = len(P)
    Wp = rng.multinomial(m, np.full(Np, 1.0 / Np), size=Bf) / m - 1.0 / Np
    KP = prob.KZZ[np.ix_(P, P)]
    return np.sqrt(np.maximum(0, np.einsum('bi,bi->b', Wp @ KP, Wp)))


def order_quantile(S, alpha):
    """inf{u : (1/B) #{S_b <= u} >= 1-alpha}  (S: ... x B)."""
    B = S.shape[-1]
    k = int(np.ceil((1 - alpha) * B)) - 1
    k = min(max(k, 0), B - 1)
    return np.sort(S, axis=-1)[..., k]


def fusion_accept(rs, q, theta):
    """accept_{bj} = 1{ D(Q_c,b^m, Q_hj) + q_bj < theta }  (rejection of the equivalence null)."""
    D = np.sqrt(np.maximum(0, np.einsum('ljj->lj', rs.G)))
    theta = np.broadcast_to(np.asarray(theta, float), (rs.G.shape[1],))
    return (D + q) < theta[None, :], D


def pool_weights(accept, ell, m):
    """sample-size pooling weights of accepted sources (row-wise)."""
    a = accept * ell[None, :]
    return a / (m + a.sum(1, keepdims=True))


# ----------------------------------------------------------------------------
# partial bootstrap causal test (Yang et al. 2026, Algorithm 2), frozen selection
# ----------------------------------------------------------------------------

def partial_bootstrap(prob, S, B, alpha, rng, rs0=None):
    """Delta = sqrt(n)[D^2(Q_f,Q_t) - D^2(Q_f,Q_c)] with Q_f = pooled control + accepted
    sources (sample-size weights). Returns (reject, Delta, crit). S: bool J vector.
    Uses observed control indices 0..m-1 and treatment m..N-1."""
    m, n = prob.m, prob.n
    ellS = prob.ell * S
    lam = m / (m + ellS.sum())
    w = ellS / (m + ellS.sum())
    # observed statistic via row-0 stats
    Kcc = prob.KZZ[:m, :m]
    c = np.full(m, 1.0 / m)
    t_idx = np.arange(m, m + n)
    cc = c @ Kcc @ c
    tt = prob.KZZ[np.ix_(t_idx, t_idx)].mean()
    ct = prob.KZZ[:m, m:].mean()
    ch = prob.r[:m].mean(0)
    th = prob.r[m:].mean(0)
    ft = lam * ct + w @ th
    fc = lam * cc + w @ ch
    delta = np.sqrt(n) * (tt - cc - 2 * ft + 2 * fc)
    # bootstrap
    Cc = rng.multinomial(m, np.full(m, 1.0 / m), size=B) / m
    Ct = rng.multinomial(n, np.full(m, 1.0 / m), size=B) / n
    CcK = Cc @ Kcc
    ccs = np.einsum('bi,bi->b', CcK, Cc)
    tts = np.einsum('bi,bi->b', Ct @ Kcc, Ct)
    cts = np.einsum('bi,bi->b', CcK, Ct)
    fts = lam * cts
    fcs = lam * ccs
    for j in range(prob.J):
        if not S[j]:
            continue
        l = prob.ell[j]
        Hj = rng.multinomial(l, np.full(l, 1.0 / l), size=B) / l
        HK = Hj @ prob.KHZ[j][:, :m]          # B x m
        fts = fts + w[j] * np.einsum('bi,bi->b', HK, Ct)
        fcs = fcs + w[j] * np.einsum('bi,bi->b', HK, Cc)
    ds = np.sqrt(n) * (tts - ccs - 2 * fts + 2 * fcs)
    crit = order_quantile(ds, alpha)
    return bool(delta > crit), float(delta), float(crit)
