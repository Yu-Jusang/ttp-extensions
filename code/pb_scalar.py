"""Gaussian limit experiment for the partial-bootstrap statistic when Q_h = Q_c = Q_t
(degenerate case), linear kernel / scalar embedding, in units where Var(X)=1 and n=1.
True:      Lam  = -2(1-g) U V + V^2,  U = h - c, V = t - c
Bootstrap: Lam* = -2(1-g)(U_obs + U*) V* + V*^2, U* = h* - c*, V* = t* - c*  (fresh, shared c*)
Reports P(Lam > q*_{1-alpha}(U_obs)) for several ratios."""
import numpy as np
rng = np.random.default_rng(20261018)
R, Bb, alpha = 20000, 4000, 0.05
for c1, c2 in [(0.5, 1.0), (0.5, 0.5), (1.0, 1.0), (0.25, 1.0), (1.0, 4.0)]:
    g = c1 / (c1 + c2)
    c = rng.normal(0, np.sqrt(1 / c1), R); t = rng.normal(0, 1, R); h = rng.normal(0, np.sqrt(1 / c2), R)
    U, V = h - c, t - c
    lam = -2 * (1 - g) * U * V + V ** 2
    cs = rng.normal(0, np.sqrt(1 / c1), (R // 10, Bb)); ts = rng.normal(0, 1, (R // 10, Bb)); hs = rng.normal(0, np.sqrt(1 / c2), (R // 10, Bb))
    Us, Vs = hs - cs, ts - cs
    lams = -2 * (1 - g) * (U[:R // 10, None] + Us) * Vs + Vs ** 2
    q = np.quantile(lams, 1 - alpha, axis=1)
    rej = np.mean(lam[:R // 10] > q)
    se = np.sqrt(rej * (1 - rej) / (R // 10))
    print(f"c1={c1} c2={c2} gamma={g:.3f}: P(reject)={rej:.4f} (MC se {se:.4f})")
