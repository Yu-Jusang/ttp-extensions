"""
Procedures compared in the two studies. Every procedure returns a p-value or a
reject flag at level alpha; all procedures in one call share the same data,
the same current-label allocations and the same bootstrap weights
(common random numbers), so paired comparisons are meaningful.
"""
import time
import numpy as np
import ttp


def _t():
    return time.perf_counter()


def study_A(prob, rng, cfg):
    """Multi-source procedures (study A). cfg keys:
    B, Bf, alpha, alpha_f (per-source fusion level), theta, grid, split_frac, Bpb"""
    out, tm, info = {}, {}, {}
    alpha = cfg['alpha']
    J, m, N = prob.J, prob.m, prob.N
    P = np.arange(N)

    t0 = _t()
    mask = ttp.allocations(rng, N, m, cfg['B'])
    rs = ttp.RowStats(prob, P, mask)
    tm['base'] = _t() - t0

    # --- current only
    t0 = _t()
    out['CUR'] = ttp.perm_pvalue(rs.Tcur)
    tm['CUR'] = _t() - t0

    # --- always pool every source (sample-size weights, fixed rule => exact)
    t0 = _t()
    wall = prob.ell / (m + prob.ell.sum())
    out['ALL'] = ttp.perm_pvalue(rs.Tbor(wall))
    tm['ALL'] = _t() - t0

    # --- risk weights, unbiased objective, full reselection
    t0 = _t()
    Wu = ttp.select_weights(rs, cfg['grid'], 'unbiased')
    Tw = rs.Tbor(Wu)
    out['WU-R'] = ttp.perm_pvalue(Tw)
    tm['WU-R'] = _t() - t0
    info['WU_w0'] = Wu[0].copy()
    info['WU_disp0'] = float(rs.disp2(Wu)[0])
    info['WU_dispbound_ok'] = bool(np.all(rs.disp2(Wu) <= 2 * Wu.sum(1) * rs.vc / rs.m + 1e-12))
    # frozen weights
    t0 = _t()
    out['WU-F'] = ttp.perm_pvalue(rs.Tbor(Wu[0]))
    tm['WU-F'] = _t() - t0 + tm['WU-R']
    # conservative (plug-in) objective, full reselection
    t0 = _t()
    Wc = ttp.select_weights(rs, cfg['grid'], 'conservative', prob.vh, prob.ell)
    out['WC-R'] = ttp.perm_pvalue(rs.Tbor(Wc))
    tm['WC-R'] = _t() - t0
    info['WC_w0'] = Wc[0].copy()

    # --- sourcewise equivalence fusion
    t0 = _t()
    dh = ttp.boot_hist_dists(prob, cfg['Bf'], rng)                     # J x Bf
    dc = ttp.boot_ctrl_dists_rows(prob, P, mask, np.arange(len(mask)), cfg['Bf'], rng)  # L x Bf
    S = dc[:, None, :] + dh[None, :, :]                                # L x J x Bf
    q = ttp.order_quantile(S, cfg['alpha_f'])                          # L x J
    acc, D = ttp.fusion_accept(rs, q, cfg['theta'])
    Wsw = ttp.pool_weights(acc, prob.ell, m)
    out['SW-R'] = ttp.perm_pvalue(rs.Tbor(Wsw))
    tm['SW-R'] = _t() - t0
    info['SW_acc0'] = acc[0].copy()
    info['SW_accrate_rows'] = acc.mean(0)
    # unadjusted fusion level (no Bonferroni)
    qu = ttp.order_quantile(S, cfg['alpha_f'] * J)
    accu, _ = ttp.fusion_accept(rs, qu, cfg['theta'])
    out['SWu-R'] = ttp.perm_pvalue(rs.Tbor(ttp.pool_weights(accu, prob.ell, m)))
    info['SWu_acc0'] = accu[0].copy()
    # frozen selection + partial permutation
    t0 = _t()
    out['SW-F'] = ttp.perm_pvalue(rs.Tbor(Wsw[0]))
    tm['SW-F'] = _t() - t0
    # frozen selection + partial bootstrap (original causal test); no source -> current permutation
    t0 = _t()
    if acc[0].any():
        rej, _, _ = ttp.partial_bootstrap(prob, acc[0], cfg['Bpb'], alpha, rng)
        out['SW-PB'] = 0.0 if rej else 1.0
    else:
        out['SW-PB'] = out['CUR']
    tm['SW-PB'] = _t() - t0

    # --- safeguards (minP across the same rows)
    t0 = _t()
    out['G-WU'] = ttp.minp_pvalue(rs.Tcur, Tw)
    out['G-SW'] = ttp.minp_pvalue(rs.Tcur, rs.Tbor(Wsw))
    tm['G'] = _t() - t0

    # --- sample splitting: pilot controls select, remaining controls test (fixed rule)
    t0 = _t()
    m1 = int(round(cfg['split_frac'] * m))
    m2 = m - m1
    pilotP = np.arange(m1)                 # observed controls are iid -> first m1 = random pilot
    pm = np.ones((1, m1 + 0), bool)
    # pilot "row": control = pilot controls, no treatment needed for selection
    rs_p = _PilotStats(prob, pilotP)
    dcp = ttp.boot_ctrl_dists_rows(prob, pilotP, np.ones((1, m1), bool), [0], cfg['Bf'], rng)
    qp = ttp.order_quantile(dcp[:, None, :] + dh[None, :, :], cfg['alpha_f'])
    accp = (np.sqrt(np.maximum(0, np.diag(rs_p.G))) + qp[0]) < cfg['theta']
    testP = np.concatenate([np.arange(m1, m), np.arange(m, N)])
    mask2 = ttp.allocations(rng, len(testP), m2, cfg['B'])
    rs2 = ttp.RowStats(prob, testP, mask2)
    w_split = (accp * prob.ell) / (m2 + (accp * prob.ell).sum())
    out['SW-S'] = ttp.perm_pvalue(rs2.Tbor(w_split))
    # weights from the pilot, applied to the test sample
    wp = _pilot_weights(rs_p, cfg['grid'])
    out['WU-S'] = ttp.perm_pvalue(rs2.Tbor(wp))
    tm['split'] = _t() - t0
    info['SWS_acc'] = accp
    return out, tm, info


class _PilotStats:
    """Shift Gram matrix of a control-only pilot sample (no treatment)."""

    def __init__(self, prob, idx):
        m = len(idx)
        Kc = prob.KZZ[np.ix_(idx, idx)]
        cc = Kc.mean()
        ch = prob.r[idx].mean(0)
        self.G = prob.HH - ch[:, None] - ch[None, :] + cc
        self.vc = (m / (m - 1.0)) * (1 - cc)
        self.m = m


def _pilot_weights(ps, grid):
    quad = np.einsum('gi,ij,gj->g', grid, ps.G, grid)
    obj = quad + (1 - 2 * grid.sum(1)) * ps.vc / ps.m
    return grid[int(np.argmin(obj))]


def study_B(prob, rng, cfg):
    """Single/multi-source radius and mass schedules (study B). cfg keys:
    B, Bf, Bpb, alpha, alpha_f, thetas (dict name->theta_n), masses (dict name->b_n),
    pb_names (list of theta names for partial bootstrap)."""
    out, tm, info = {}, {}, {}
    alpha = cfg['alpha']
    J, m, N = prob.J, prob.m, prob.N
    P = np.arange(N)
    t0 = _t()
    mask = ttp.allocations(rng, N, m, cfg['B'])
    rs = ttp.RowStats(prob, P, mask)
    tm['base'] = _t() - t0
    out['CUR'] = ttp.perm_pvalue(rs.Tcur)
    wall = prob.ell / (m + prob.ell.sum())
    out['ALL'] = ttp.perm_pvalue(rs.Tbor(wall))

    t0 = _t()
    dh = ttp.boot_hist_dists(prob, cfg['Bf'], rng)                        # J x Bf
    dpool = ttp.boot_ctrl_dists_pooled(prob, P, m, cfg['Bf'], rng)        # Bf  (label invariant)
    dc0 = ttp.boot_ctrl_dists_rows(prob, P, mask, [0], cfg['Bf'], rng)[0]  # Bf  (observed control)
    q_pool = ttp.order_quantile(dpool[None, :] + dh, cfg['alpha_f'])      # J
    q_obs = ttp.order_quantile(dc0[None, :] + dh, cfg['alpha_f'])         # J
    Dall = np.sqrt(np.maximum(0, np.einsum('ljj->lj', rs.G)))             # L x J
    tm['fusion_boot'] = _t() - t0
    info['D0'] = Dall[0].copy()
    info['q_obs'] = q_obs.copy()

    for name, th in cfg['thetas'].items():
        t0 = _t()
        accR = (Dall + q_pool[None, :]) < th
        TR = rs.Tbor(ttp.pool_weights(accR, prob.ell, m))
        out[f'R-{name}'] = ttp.perm_pvalue(TR)
        info[f'accR-{name}'] = accR[0].copy()
        tm[f'R-{name}'] = _t() - t0
        t0 = _t()
        accF = (Dall[0] + q_obs) < th
        wF = (accF * prob.ell) / (m + (accF * prob.ell).sum())
        out[f'F-{name}'] = ttp.perm_pvalue(rs.Tbor(wF))
        info[f'accF-{name}'] = accF.copy()
        tm[f'F-{name}'] = _t() - t0
        if name in cfg.get('guard_names', []):
            out[f'G-{name}'] = ttp.minp_pvalue(rs.Tcur, TR)
        if name in cfg.get('pb_names', []):
            t0 = _t()
            if accF.any():
                rej, _, _ = ttp.partial_bootstrap(prob, accF, cfg['Bpb'], alpha, rng)
                out[f'PB-{name}'] = 0.0 if rej else 1.0
            else:
                out[f'PB-{name}'] = out['CUR']
            tm[f'PB-{name}'] = _t() - t0

    for name, b in cfg.get('masses', {}).items():
        w = np.full(J, b / J)
        out[f'V-{name}'] = ttp.perm_pvalue(rs.Tbor(w))

    if cfg.get('pb_always', False):
        t0 = _t()
        rej, _, _ = ttp.partial_bootstrap(prob, np.ones(J, bool), cfg['Bpb'], alpha, rng)
        out['PB-ALL'] = 0.0 if rej else 1.0
        tm['PB-ALL'] = _t() - t0
    return out, tm, info
