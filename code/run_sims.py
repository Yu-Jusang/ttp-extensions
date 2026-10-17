"""
Run all simulation suites. Usage:
    python run_sims.py <suite> [reps_null] [reps_alt] [seed]
Suites: A_main, A_n, B_n, B_local, repro, plus *_conf confirmation variants
(confirmation = identical design, independent seed). Results -> ../results/<suite>.npz
"""
import sys, os, time, json, platform
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np
from multiprocessing import Pool
import ttp, methods

ALPHA = 0.05
NORM = lambda mu, sd=1.0: ('norm', mu, sd)
SHAPE = ('mix', [(0.5, -0.8, 0.6), (0.5, 0.8, 0.6)])     # mean 0, variance 1, bimodal

# ------------------------------------------------------------------ study A
A_SCEN = {
    'Aligned':  [NORM(0), NORM(0), NORM(0)],
    'Partial':  [NORM(0), NORM(0), NORM(1.0)],
    'Shifted':  [NORM(0.5), NORM(0.7), NORM(1.0)],
    'Opposite': [NORM(0.6), NORM(-0.6), NORM(1.5)],
    'Mixed':    [NORM(0), NORM(0, 1.5), NORM(0.8)],
}
A_TRT = {
    'Null': NORM(0), 'Mean+': NORM(0.5), 'Mean-': NORM(-0.5),
    'Scale': NORM(0, 1.5), 'Shape': SHAPE,
}


def a_cfg(m):
    return dict(B=199, Bf=199, Bpb=199, alpha=ALPHA, alpha_f=ALPHA / 3, theta=0.5,
                grid=ttp.simplex_grid(3, 0.05, 0.9), split_frac=0.5)


def task_A(args):
    key, m, n, ell, cspec, tspec, hspecs, seed = args
    rng = np.random.default_rng(seed)
    xc = ttp.sample(rng, cspec, m)
    xt = ttp.sample(rng, tspec, n)
    xh = [ttp.sample(rng, h, ell) for h in hspecs]
    prob = ttp.Problem(xc, xt, xh, 1.0)
    out, tm, info = methods.study_A(prob, rng, a_cfg(m))
    return key, out, tm, info


def suite_A_main(reps0, reps1, seed):
    tasks = []
    m, n, ell = 40, 80, 80
    for si, (s, hs) in enumerate(A_SCEN.items()):
        for ti, (t, ts) in enumerate(A_TRT.items()):
            R = reps0 if t == 'Null' else reps1
            for r in range(R):
                ss = np.random.SeedSequence([seed, si, ti, r]).generate_state(1)[0]
                tasks.append(((s, t, m), m, n, ell, NORM(0), ts, hs, int(ss)))
    return task_A, tasks


def suite_A_n(reps0, reps1, seed):
    # blind spot: all sources N(0.6,1) accepted at theta=0.5 for moderate m;
    # treatment = sample-size mixture so that pooling all sources reproduces Q_t
    lam = 1.0 / 7.0          # m/(m+3*ell) with ell = 2m
    settings = {
        'AlignedNull': ([NORM(0)] * 3, NORM(0)),
        'AlignedMean': ([NORM(0)] * 3, NORM(0.4)),
        'ShiftedNull': ([NORM(0.6)] * 3, NORM(0)),
        'BlindSpot':   ([NORM(0.6)] * 3, ('mix', [(lam, 0, 1), (1 - lam, 0.6, 1)])),
    }
    tasks = []
    for si, (s, (hs, ts)) in enumerate(settings.items()):
        for mi, m in enumerate([20, 40, 80, 160]):
            R = reps0 if 'Null' in s else reps1
            for r in range(R):
                ss = np.random.SeedSequence([seed, 100 + si, mi, r]).generate_state(1)[0]
                tasks.append(((s, 'n', m), m, 2 * m, 2 * m, NORM(0), ts, hs, int(ss)))
    return task_A, tasks


# ------------------------------------------------------------------ study B
N0, THETA0 = 50, 0.5
EXPS = {'a25': 0.25, 'a45': 0.45, 'a75': 0.75}


def b_cfg(n, J):
    th = {'fix': THETA0}
    for k, a in EXPS.items():
        th[k] = THETA0 * (n / N0) ** (-a)
    masses = {'b25': (2 / 3) * (n / N0) ** (-0.25), 'b50': (2 / 3) * (n / N0) ** (-0.5)}
    return dict(B=199, Bf=199, Bpb=199, alpha=ALPHA, alpha_f=ALPHA / J, thetas=th,
                masses=masses, pb_names=['fix', 'a25'], guard_names=['a25', 'fix'],
                pb_always=True)


def task_B(args):
    key, n, cspec, tspec, hspecs, seed = args
    rng = np.random.default_rng(seed)
    m, ell = n // 2, n
    xc = ttp.sample(rng, cspec, m)
    xt = ttp.sample(rng, tspec, n)
    xh = [ttp.sample(rng, h, ell) for h in hspecs]
    prob = ttp.Problem(xc, xt, xh, 1.0)
    out, tm, info = methods.study_B(prob, rng, b_cfg(n, len(hspecs)))
    return key, out, tm, info


NS = [50, 100, 200, 400, 800]
GAM = 1.0 / 3.0       # m/(m+ell) with m=n/2, ell=n


def suite_B_n(reps0, reps1, seed):
    mu_bs = 0.8       # D(Qc,Qh)=0.342 < theta_fix=0.5
    settings = {
        'NullIdent':  ([NORM(0)], NORM(0)),
        'NullShift':  ([NORM(0.3)], NORM(0)),
        'NullFar':    ([NORM(1.0)], NORM(0)),
        'PowIdent':   ([NORM(0)], NORM(0.3)),
        'BlindSpot':  ([NORM(mu_bs)], ('mix', [(GAM, 0, 1), (1 - GAM, mu_bs, 1)])),
        'Multi2Null': ([NORM(0), NORM(0.8)], NORM(0)),
        'Multi2Pow':  ([NORM(0), NORM(0.8)], NORM(0.3)),
    }
    tasks = []
    for si, (s, (hs, ts)) in enumerate(settings.items()):
        for ni, n in enumerate(NS):
            R = reps0 if 'Null' in s else reps1
            for r in range(R):
                ss = np.random.SeedSequence([seed, 200 + si, ni, r]).generate_state(1)[0]
                tasks.append(((s, 'n', n), n, NORM(0), ts, hs, int(ss)))
    return task_B, tasks


def suite_B_local(reps0, reps1, seed):
    """sample-size-dependent populations (triangular arrays)."""
    tasks = []
    for ni, n in enumerate(NS):
        th25 = THETA0 * (n / N0) ** (-0.25)
        # (i) historical discrepancy at half the a25 radius; treatment = pooled mixture
        mu_h = ttp.mu_for_mmd(0.5 * th25)
        loc_bs = ([NORM(mu_h)], ('mix', [(GAM, 0, 1), (1 - GAM, mu_h, 1)]))
        # (ii) boundary null: discrepancy equal to the a25 radius
        mu_b = ttp.mu_for_mmd(th25)
        bnull = ([NORM(mu_b)], NORM(0))
        # (iii) root-n local mean alternative with an identical source
        loc_pow = ([NORM(0)], NORM(2.5 / np.sqrt(n)))
        # (iv) root-n local alternative opposed to a root-n historical shift
        mu_hl = 2.0 / np.sqrt(n)
        loc_opp = ([NORM(mu_hl)], NORM(2.5 / np.sqrt(n)))
        for si, (s, (hs, ts)) in enumerate({'LocBlind': loc_bs, 'BoundNull': bnull,
                                             'LocPow': loc_pow, 'LocAlign': loc_opp}.items()):
            R = reps0 if 'Null' in s else reps1
            for r in range(R):
                ss = np.random.SeedSequence([seed, 300 + si, ni, r]).generate_state(1)[0]
                tasks.append(((s, 'n', n), n, NORM(0), ts, hs, int(ss)))
    return task_B, tasks


# ------------------------------------------------------------------ reproduction
def task_repro(args):
    key, dmu, seed = args[:3]
    bw = args[3] if len(args) > 3 else None
    rng = np.random.default_rng(seed)
    m, n, ell = 50, 100, 100
    xt = rng.normal(0, 1, n)
    xc = rng.normal(dmu, 1, m)
    xh = rng.normal(dmu + 0.2, 1, ell)
    sig = ttp.median_heuristic_sigma(np.concatenate([xc, xt, xh])) if bw is None else bw
    prob = ttp.Problem(xc, xt, [xh], sig)
    cfg = dict(B=999, Bf=999, Bpb=999, alpha=ALPHA, alpha_f=ALPHA, thetas={'fix': 0.4},
               masses={}, pb_names=['fix'], guard_names=['fix'], pb_always=False)
    out, tm, info = methods.study_B(prob, rng, cfg)
    return key, out, tm, info


def suite_repro(reps0, reps1, seed):
    tasks = []
    for di, d in enumerate([-0.4, 0.0, 0.4]):
        R = reps0 if d == 0 else reps1
        for r in range(R):
            ss = np.random.SeedSequence([seed, 400 + di, r]).generate_state(1)[0]
            tasks.append(((f'{d:+.1f}', 'repro', 0), d, int(ss)))
    return task_repro, tasks


def suite_repro_bw(reps0, reps1, seed):
    """fixed bandwidths s in {0.5,1,1.5} read as k=exp(-d^2/s^2), i.e. sigma=s/sqrt(2)."""
    tasks = []
    for si, s in enumerate([0.5, 1.0, 1.5]):
        for di, d in enumerate([-0.4, 0.0, 0.4]):
            R = reps0 if d == 0 else reps1
            for r in range(R):
                ss = np.random.SeedSequence([seed, 500 + si, di, r]).generate_state(1)[0]
                tasks.append(((f'{d:+.1f}', f'bw{s}', 0), d, int(ss), s / np.sqrt(2)))
    return task_repro, tasks


SUITES = {'A_main': suite_A_main, 'A_n': suite_A_n, 'B_n': suite_B_n,
          'B_local': suite_B_local, 'repro': suite_repro, 'repro_bw': suite_repro_bw}


def main():
    suite = sys.argv[1]
    reps0 = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    reps1 = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 20261018
    base = suite.replace('_conf', '')
    fn, tasks = SUITES[base](reps0, reps1, seed)
    t0 = time.time()
    res = {}
    with Pool(15) as pool:
        for key, out, tm, info in pool.imap_unordered(fn, tasks, chunksize=4):
            d = res.setdefault(key, {'p': {}, 't': {}, 'info': {}})
            for k, v in out.items():
                d['p'].setdefault(k, []).append(v)
            for k, v in tm.items():
                d['t'].setdefault(k, []).append(v)
            for k, v in info.items():
                d['info'].setdefault(k, []).append(np.asarray(v, float))
    elapsed = time.time() - t0
    outdir = os.path.join(os.path.dirname(__file__), '..', 'results')
    flat = {}
    for key, d in res.items():
        tag = '|'.join(str(x) for x in key)
        for k, v in d['p'].items():
            flat[f'p|{tag}|{k}'] = np.array(v)
        for k, v in d['t'].items():
            flat[f't|{tag}|{k}'] = np.array(v)
        for k, v in d['info'].items():
            flat[f'i|{tag}|{k}'] = np.array(v)
    np.savez_compressed(os.path.join(outdir, f'{suite}.npz'), **flat)
    meta = dict(suite=suite, reps_null=reps0, reps_alt=reps1, seed=seed, n_tasks=len(tasks),
                elapsed_sec=elapsed, python=platform.python_version(), numpy=np.__version__,
                platform=platform.platform(), processor=platform.processor(), workers=15)
    with open(os.path.join(outdir, f'{suite}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(json.dumps(meta))


if __name__ == '__main__':
    main()
