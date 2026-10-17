"""
Tables (LaTeX fragments) and black-and-white figures from the raw Monte Carlo output.
Usage: python analyze.py   (reads ../results/*.npz, writes ../paper_A/gen, ../paper_B/gen)
"""
import os, json, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, '..', 'results')
GA = os.path.join(HERE, '..', 'paper_A', 'gen')
GB = os.path.join(HERE, '..', 'paper_B', 'gen')
for d in (GA, GB):
    os.makedirs(d, exist_ok=True)
ALPHA = 0.05

plt.rcParams.update({'font.family': 'serif', 'font.size': 8.5, 'axes.linewidth': 0.6,
                     'lines.linewidth': 1.0, 'lines.markersize': 3.6, 'legend.fontsize': 7,
                     'legend.frameon': False, 'axes.spines.top': False, 'axes.spines.right': False,
                     'xtick.major.width': 0.5, 'ytick.major.width': 0.5, 'savefig.bbox': 'tight',
                     'pdf.fonttype': 42})
STYLES = [('-', 'o', '0.0'), ('--', 's', '0.0'), (':', '^', '0.0'), ('-.', 'D', '0.0'),
          ('-', 'v', '0.45'), ('--', 'x', '0.45'), (':', '+', '0.45'), ('-.', '*', '0.45'),
          ('-', 'P', '0.7'), ('--', 'h', '0.7')]


def load(name):
    path = os.path.join(RES, name + '.npz')
    if not os.path.exists(path):
        return None
    z = np.load(path)
    P, T, I = {}, {}, {}
    for k in z.files:
        parts = k.split('|')
        kind, key, meth = parts[0], tuple(parts[1:-1]), parts[-1]
        {'p': P, 't': T, 'i': I}[kind].setdefault(key, {})[meth] = z[k]
    return P, T, I


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def rate(p, alpha=ALPHA):
    r = np.asarray(p) <= alpha + 1e-12
    return wilson(r.sum(), len(r))


def fmt(x, nd=3):
    return f'{x:.{nd}f}'


def cell(p, alpha=ALPHA, null=False):
    """rate with a marker when the Wilson interval lies above alpha (null rows only)."""
    r, lo, hi = rate(p, alpha)
    s = fmt(r)
    if null and lo > alpha:
        s = r'\textbf{' + s + r'}$^{\uparrow}$'
    return s


def write(path, txt):
    with io.open(path, 'w', encoding='utf-8') as f:
        f.write(txt)


# --------------------------------------------------------------------- repro
def repro_table():
    d = load('repro')
    if d is None:
        return
    P, T, I = d
    rows = []
    order = ['-0.4', '+0.0', '+0.4']
    for dm in order:
        key = (dm, 'repro', '0')
        p = P[key]
        acc = I[key]['accF-fix'].mean()
        rows.append((dm, acc, p))
    cols = [('CUR', 'Current only'), ('PB-fix', 'PB (orig.)'), ('F-fix', 'PP frozen (orig.)'),
            ('R-fix', 'PP reselect'), ('ALL', 'Always pool'), ('G-fix', 'minP guard')]
    s = [r'\begin{tabular}{@{}lr' + 'r' * len(cols) + r'@{}}', r'\toprule',
         r'$\mu_c-\mu_t$ & merge & ' + ' & '.join(c[1] for c in cols) + r'\\', r'\midrule']
    for dm, acc, p in rows:
        null = dm == '+0.0'
        s.append(f'{float(dm):+.1f} & {acc:.3f} & ' + ' & '.join(cell(p[c[0]], null=null) for c in cols) + r'\\')
    s += [r'\midrule', r'\multicolumn{%d}{@{}l}{\footnotesize Original Table 2 (median heuristic): merge 0.285/0.307/0.272; PB 0.577/0.064/0.572; PP 0.513/0.051/0.535.}\\' % (len(cols) + 2),
          r'\bottomrule', r'\end{tabular}']
    write(os.path.join(GA, 'tab_repro.tex'), '\n'.join(s))
    write(os.path.join(GB, 'tab_repro.tex'), '\n'.join(s))


# --------------------------------------------------------------------- study A
A_SCEN = ['Aligned', 'Partial', 'Shifted', 'Opposite', 'Mixed']
A_TRT = ['Null', 'Mean+', 'Mean-', 'Scale', 'Shape']
A_COLS = [('CUR', 'CUR'), ('ALL', 'ALL'), ('WU-R', 'WU-R'), ('WU-F', 'WU-F'), ('WC-R', 'WC-R'),
          ('SW-R', 'SW-R'), ('SW-F', 'SW-F'), ('SW-PB', 'SW-PB'), ('SW-S', 'SW-S'), ('WU-S', 'WU-S'),
          ('G-WU', 'G-WU'), ('G-SW', 'G-SW')]


def table_A_main(suite='A_main', tag=''):
    d = load(suite)
    if d is None:
        return None
    P, T, I = d
    out = {}
    # Type-I table
    s = [r'\begin{tabular}{@{}l' + 'r' * len(A_COLS) + r'@{}}', r'\toprule',
         'Sources & ' + ' & '.join(c[1] for c in A_COLS) + r'\\', r'\midrule']
    for sc in A_SCEN:
        key = (sc, 'Null', '40')
        if key not in P:
            continue
        s.append(sc + ' & ' + ' & '.join(cell(P[key][c[0]], null=True) for c in A_COLS) + r'\\')
    s += [r'\bottomrule', r'\end{tabular}']
    write(os.path.join(GA, f'tab_A_null{tag}.tex'), '\n'.join(s))
    # power table (one block per treatment)
    s = [r'\begin{tabular}{@{}ll' + 'r' * len(A_COLS) + r'@{}}', r'\toprule',
         r'Treatment & Sources & ' + ' & '.join(c[1] for c in A_COLS) + r'\\', r'\midrule']
    for tr in A_TRT[1:]:
        first = True
        for sc in A_SCEN:
            key = (sc, tr, '40')
            if key not in P:
                continue
            rates = [rate(P[key][c[0]])[0] for c in A_COLS]
            base = rates[0]
            cells = []
            for c, r in zip(A_COLS, rates):
                x = fmt(r)
                if c[0] != 'CUR' and r >= base + 0.03:
                    x = r'\textbf{' + x + '}'
                elif c[0] != 'CUR' and r <= base - 0.03:
                    x = r'\textit{' + x + '}'
                cells.append(x)
            s.append((tr if first else '') + ' & ' + sc + ' & ' + ' & '.join(cells) + r'\\')
            first = False
        s.append(r'\addlinespace[2pt]')
    s[-1] = r'\bottomrule'
    s.append(r'\end{tabular}')
    write(os.path.join(GA, f'tab_A_power{tag}.tex'), '\n'.join(s))
    # weights / acceptance table
    s = [r'\begin{tabular}{@{}llrrrrrrrrr@{}}', r'\toprule',
         r' & & \multicolumn{3}{c}{mean $\hat w_j$ (WU)} & \multicolumn{3}{c}{mean $\hat w_j$ (WC)} & \multicolumn{3}{c}{accept rate (SW)}\\',
         r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}',
         r'Sources & Treatment & $h_1$ & $h_2$ & $h_3$ & $h_1$ & $h_2$ & $h_3$ & $h_1$ & $h_2$ & $h_3$\\', r'\midrule']
    for sc in A_SCEN:
        for tr in ['Null', 'Mean+']:
            key = (sc, tr, '40')
            if key not in I:
                continue
            wu = I[key]['WU_w0'].mean(0)
            wc = I[key]['WC_w0'].mean(0)
            ac = I[key]['SW_acc0'].mean(0)
            s.append(f'{sc} & {tr} & ' + ' & '.join(fmt(x, 2) for x in np.r_[wu, wc, ac]) + r'\\')
    s += [r'\bottomrule', r'\end{tabular}']
    write(os.path.join(GA, f'tab_A_weights{tag}.tex'), '\n'.join(s))
    # paired power differences vs CUR (figure)
    meths = ['ALL', 'WU-R', 'WC-R', 'SW-R', 'SW-S', 'G-WU', 'G-SW']
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.7), sharey=True)
    for ax, tr in zip(axes, A_TRT[1:]):
        for k, mth in enumerate(meths):
            xs, ys, es = [], [], []
            for i, sc in enumerate(A_SCEN):
                key = (sc, tr, '40')
                if key not in P:
                    continue
                a = (P[key][mth] <= ALPHA).astype(float)
                b = (P[key]['CUR'] <= ALPHA).astype(float)
                dlt = a - b
                xs.append(i + (k - 3) * 0.1)
                ys.append(dlt.mean())
                es.append(1.96 * dlt.std(ddof=1) / np.sqrt(len(dlt)))
            ls, mk, col = STYLES[k]
            ax.errorbar(xs, ys, yerr=es, fmt=mk, color=col, ms=3.2, elinewidth=0.6, capsize=1.2,
                        label=mth, mfc='white' if k % 2 else col)
        ax.axhline(0, color='0.3', lw=0.5)
        ax.set_xticks(range(len(A_SCEN)))
        ax.set_xticklabels([s[:4] for s in A_SCEN], rotation=0)
        ax.set_title(tr, fontsize=8.5)
    axes[0].set_ylabel('power $-$ power(CUR)')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.savefig(os.path.join(GA, f'fig_A_diff{tag}.pdf'))
    plt.close(fig)
    # timing
    tsum = {}
    for key, tt in T.items():
        for mth, v in tt.items():
            tsum.setdefault(mth, []).append(v)
    out['timing'] = {k: float(np.mean(np.concatenate(v))) for k, v in tsum.items()}
    return out


def tables_A_extra(suite='A_main'):
    d = load(suite)
    if d is None:
        return
    P, T, I = d
    # Bonferroni vs unadjusted fusion levels, and observed-vs-permuted acceptance
    s = [r'\begin{tabular}{@{}llrrrr@{}}', r'\toprule',
         r'Sources & Treatment & SW-R & SWu-R & accept (obs.\ row) $h_1,h_2,h_3$ & accept (perm.\ rows) $h_1,h_2,h_3$\\',
         r'\midrule']
    for sc in A_SCEN:
        for tr in A_TRT:
            key = (sc, tr, '40')
            if key not in P:
                continue
            a0 = I[key]['SW_acc0'].mean(0)
            ar = I[key]['SW_accrate_rows'].mean(0)
            s.append(f'{sc} & {tr} & ' + cell(P[key]['SW-R'], null=tr == 'Null') + ' & ' +
                     cell(P[key]['SWu-R'], null=tr == 'Null') + ' & ' + ' '.join(fmt(x, 2) for x in a0) +
                     ' & ' + ' '.join(fmt(x, 2) for x in ar) + r'\\')
        s.append(r'\addlinespace[2pt]')
    s[-1] = r'\bottomrule'
    s.append(r'\end{tabular}')
    write(os.path.join(GA, 'tab_A_accrows.tex'), '\n'.join(s))


def table_repro_bw():
    d = load('repro_bw')
    if d is None:
        return
    P, T, I = d
    orig = {'0.5': ('0.34', '0.528 (0.511)', '0.070 (0.063)', '0.536 (0.566)'),
            '1.0': ('0.67', '0.572 (0.559)', '0.074 (0.053)', '0.711 (0.629)'),
            '1.5': ('0.82', '0.581 (0.566)', '0.074 (0.065)', '0.739 (0.677)')}
    s = [r'\begin{tabular}{@{}lrrrrrrr@{}}', r'\toprule',
         r'$s$ & merge & \multicolumn{2}{c}{$\mu_c-\mu_t=-0.4$} & \multicolumn{2}{c}{$\mu_c=\mu_t$} & \multicolumn{2}{c}{$\mu_c-\mu_t=+0.4$}\\',
         r'\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}',
         r' & & PB & PP & PB & PP & PB & PP\\', r'\midrule']
    for bw in ['0.5', '1.0', '1.5']:
        cells, merges = [], []
        for dm in ['-0.4', '+0.0', '+0.4']:
            key = (dm, f'bw{bw}', '0')
            if key not in P:
                cells += ['--', '--']
                continue
            merges.append(I[key]['accF-fix'].mean())
            cells += [cell(P[key]['PB-fix'], null=dm == '+0.0'), cell(P[key]['F-fix'], null=dm == '+0.0')]
        mg = fmt(np.mean(merges)) if merges else '--'
        s.append(f'{bw} & {mg} & ' + ' & '.join(cells) + r'\\')
        o = orig[bw]
        s.append(r'\quad original & ' + o[0] + ' & ' + ' & '.join(x.replace(' (', ' & ').replace(')', '') for x in o[1:]) + r'\\')
    s += [r'\bottomrule', r'\end{tabular}']
    write(os.path.join(GA, 'tab_repro_bw.tex'), '\n'.join(s))
    write(os.path.join(GB, 'tab_repro_bw.tex'), '\n'.join(s))


A_N_SETTINGS = ['AlignedNull', 'ShiftedNull', 'AlignedMean', 'BlindSpot']


def fig_A_n(suite='A_n', tag=''):
    d = load(suite)
    if d is None:
        return None
    P, T, I = d
    meths = [('CUR', 'CUR'), ('ALL', 'ALL'), ('WU-R', 'WU-R'), ('WU-F', 'WU-F'), ('SW-R', 'SW-R'),
             ('SW-F', 'SW-F'), ('SW-PB', 'SW-PB'), ('G-SW', 'G-SW')]
    ms = [20, 40, 80, 160]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.5))
    for ax, st in zip(axes, A_N_SETTINGS):
        for k, (mth, lab) in enumerate(meths):
            r = [rate(P[(st, 'n', str(m))][mth]) for m in ms if (st, 'n', str(m)) in P]
            if not r:
                continue
            r = np.array(r)
            ls, mk, col = STYLES[k]
            ax.errorbar(ms[:len(r)], r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], ls=ls, marker=mk,
                        color=col, mfc='white' if k % 2 else col, elinewidth=0.5, capsize=1.0, label=lab)
        ax.set_xscale('log', base=2)
        ax.set_xticks(ms)
        ax.set_xticklabels([str(m) for m in ms])
        ax.set_xlabel('$m$')
        ax.set_title(st, fontsize=8.5)
        if 'Null' in st:
            ax.axhline(ALPHA, color='0.5', lw=0.5, ls=':')
    axes[0].set_ylabel('rejection rate')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout()
    fig.savefig(os.path.join(GA, f'fig_A_n{tag}.pdf'))
    plt.close(fig)
    # table
    s = [r'\begin{tabular}{@{}lr' + 'r' * len(meths) + r'@{}}', r'\toprule',
         r'Setting & $m$ & ' + ' & '.join(m[1] for m in meths) + r'\\', r'\midrule']
    for st in A_N_SETTINGS:
        for m in ms:
            key = (st, 'n', str(m))
            if key not in P:
                continue
            s.append((st if m == 20 else '') + f' & {m} & ' +
                     ' & '.join(cell(P[key][mm[0]], null='Null' in st) for mm in meths) + r'\\')
        s.append(r'\addlinespace[2pt]')
    s[-1] = r'\bottomrule'
    s.append(r'\end{tabular}')
    write(os.path.join(GA, f'tab_A_n{tag}.tex'), '\n'.join(s))
    tim = {}
    for m in ms:
        for st in A_N_SETTINGS:
            key = (st, 'n', str(m))
            if key in T:
                for mth, v in T[key].items():
                    tim.setdefault(m, {}).setdefault(mth, []).append(v)
    return {m: {k: float(np.mean(np.concatenate(v))) for k, v in dd.items()} for m, dd in tim.items()}


# --------------------------------------------------------------------- study B
NS = [50, 100, 200, 400, 800]
B_METHS = [('CUR', 'CUR'), ('ALL', 'ALL'), ('R-fix', 'R fixed'), ('R-a25', 'R a=.25'), ('R-a45', 'R a=.45'),
           ('R-a75', 'R a=.75'), ('F-fix', 'F fixed'), ('F-a25', 'F a=.25'), ('V-b25', 'V b=.25'),
           ('G-a25', 'G a=.25')]


def B_panels(P, settings, fname, meths=B_METHS, ncol=4, null_like=()):
    rows = int(np.ceil(len(settings) / ncol))
    fig, axes = plt.subplots(rows, ncol, figsize=(7.4, 2.35 * rows), squeeze=False)
    for ax, st in zip(axes.ravel(), settings):
        for k, (mth, lab) in enumerate(meths):
            pts = [(n, rate(P[(st, 'n', str(n))][mth])) for n in NS
                   if (st, 'n', str(n)) in P and mth in P[(st, 'n', str(n))]]
            if not pts:
                continue
            n_ = [p[0] for p in pts]
            r = np.array([p[1] for p in pts])
            ls, mk, col = STYLES[k % len(STYLES)]
            ax.errorbar(n_, r[:, 0], yerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]], ls=ls, marker=mk, color=col,
                        mfc='white' if k % 2 else col, elinewidth=0.5, capsize=1.0, label=lab)
        ax.set_xscale('log', base=2)
        ax.set_xticks(NS)
        ax.set_xticklabels([str(n) for n in NS])
        ax.set_xlabel('$n$')
        ax.set_title(st, fontsize=8.5)
        if st in null_like or 'Null' in st:
            ax.axhline(ALPHA, color='0.5', lw=0.5, ls=':')
    for ax in axes.ravel()[len(settings):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('rejection rate')
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.02 - 0.03 * (rows == 1)))
    fig.tight_layout(rect=(0, 0.08 if rows > 1 else 0.13, 1, 1))
    fig.savefig(os.path.join(GB, fname))
    plt.close(fig)


def B_table(P, settings, meths, fname, null_like=()):
    s = [r'\begin{tabular}{@{}lr' + 'r' * len(meths) + r'@{}}', r'\toprule',
         r'Setting & $n$ & ' + ' & '.join(m[1] for m in meths) + r'\\', r'\midrule']
    for st in settings:
        first = True
        for n in NS:
            key = (st, 'n', str(n))
            if key not in P:
                continue
            nl = st in null_like or 'Null' in st
            s.append((st if first else '') + f' & {n} & ' +
                     ' & '.join(cell(P[key][m[0]], null=nl) if m[0] in P[key] else '--' for m in meths) + r'\\')
            first = False
        s.append(r'\addlinespace[2pt]')
    s[-1] = r'\bottomrule'
    s.append(r'\end{tabular}')
    write(os.path.join(GB, fname), '\n'.join(s))


def study_B(suite='B_n', local='B_local', tag=''):
    d = load(suite)
    res = {}
    if d is not None:
        P, T, I = d
        nulls = ['NullIdent', 'NullShift', 'NullFar', 'Multi2Null']
        alts = ['PowIdent', 'BlindSpot', 'Multi2Pow']
        B_panels(P, nulls, f'fig_B_null{tag}.pdf')
        B_panels(P, alts, f'fig_B_power{tag}.pdf', ncol=3)
        tm = [('CUR', 'CUR'), ('R-fix', 'R fix'), ('R-a25', 'R .25'), ('R-a45', 'R .45'), ('R-a75', 'R .75'),
              ('F-fix', 'F fix'), ('F-a25', 'F .25'), ('PB-fix', 'PB fix'), ('PB-a25', 'PB .25'),
              ('PB-ALL', 'PB all'), ('V-b25', 'V .25'), ('V-b50', 'V .50'), ('G-a25', 'G .25')]
        B_table(P, nulls, tm, f'tab_B_null{tag}.tex')
        tm2 = [('CUR', 'CUR'), ('ALL', 'ALL'), ('R-fix', 'R fix'), ('R-a25', 'R .25'), ('R-a45', 'R .45'),
               ('R-a75', 'R .75'), ('F-a25', 'F .25'), ('PB-a25', 'PB .25'), ('V-b25', 'V .25'),
               ('V-b50', 'V .50'), ('G-a25', 'G .25'), ('G-fix', 'G fix')]
        B_table(P, alts, tm2, f'tab_B_power{tag}.tex')
        # acceptance rates of the observed row
        fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3))
        for ax, st in zip(axes, ['NullIdent', 'NullShift', 'BlindSpot', 'Multi2Null']):
            for k, nm in enumerate(['fix', 'a25', 'a45', 'a75']):
                for src in range(2 if 'Multi2' in st else 1):
                    ys = []
                    for n in NS:
                        key = (st, 'n', str(n))
                        if key in I:
                            ys.append(I[key][f'accR-{nm}'][:, src].mean())
                    ls, mk, col = STYLES[k]
                    ax.plot(NS[:len(ys)], ys, ls=ls if src == 0 else ':', marker=mk, color=col,
                            mfc='white' if src else col, label=f'{nm}' + (f' $h_{src + 1}$' if 'Multi2' in st else ''))
            ax.set_xscale('log', base=2)
            ax.set_xticks(NS)
            ax.set_xticklabels([str(n) for n in NS])
            ax.set_ylim(-0.03, 1.03)
            ax.set_xlabel('$n$')
            ax.set_title(st, fontsize=8.5)
        axes[0].set_ylabel('acceptance rate (observed row)')
        axes[-1].legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
        fig.tight_layout()
        fig.savefig(os.path.join(GB, f'fig_B_accept{tag}.pdf'))
        plt.close(fig)
        tim = {}
        for n in NS:
            for st in nulls + alts:
                key = (st, 'n', str(n))
                if key in T:
                    for mth, v in T[key].items():
                        tim.setdefault(n, {}).setdefault(mth, []).append(v)
        res['timing'] = {n: {k: float(np.mean(np.concatenate(v))) for k, v in dd.items()} for n, dd in tim.items()}
        res['accept'] = {st: {nm: [float(I[(st, 'n', str(n))][f'accR-{nm}'].mean(0)[0]) for n in NS if (st, 'n', str(n)) in I]
                              for nm in ['fix', 'a25', 'a45', 'a75']} for st in ['NullIdent', 'NullShift', 'BlindSpot', 'PowIdent']}
    d = load(local)
    if d is not None:
        P, T, I = d
        sets = ['BoundNull', 'LocBlind', 'LocPow', 'LocAlign']
        B_panels(P, sets, f'fig_B_local{tag}.pdf', null_like=('BoundNull',))
        tm = [('CUR', 'CUR'), ('ALL', 'ALL'), ('R-fix', 'R fix'), ('R-a25', 'R .25'), ('R-a45', 'R .45'),
              ('R-a75', 'R .75'), ('F-fix', 'F fix'), ('F-a25', 'F .25'), ('PB-a25', 'PB .25'), ('V-b25', 'V .25'),
              ('V-b50', 'V .50'), ('G-a25', 'G .25')]
        B_table(P, sets, tm, f'tab_B_local{tag}.tex', null_like=('BoundNull',))
        res['accept_local'] = {st: {nm: [float(I[(st, 'n', str(n))][f'accF-{nm}'].mean()) for n in NS if (st, 'n', str(n)) in I]
                                    for nm in ['fix', 'a25', 'a45', 'a75']} for st in sets}
    return res


def main():
    summary = {}
    repro_table()
    table_repro_bw()
    tables_A_extra()
    summary['A_main'] = table_A_main()
    summary['A_n_timing'] = fig_A_n()
    summary['B'] = study_B()
    if os.path.exists(os.path.join(RES, 'A_main_conf.npz')):
        summary['A_main_conf'] = table_A_main('A_main_conf', '_conf')
    if os.path.exists(os.path.join(RES, 'A_n_conf.npz')):
        fig_A_n('A_n_conf', '_conf')
    if os.path.exists(os.path.join(RES, 'B_n_conf.npz')) or os.path.exists(os.path.join(RES, 'B_local_conf.npz')):
        summary['B_conf'] = study_B('B_n_conf', 'B_local_conf', '_conf')
    with io.open(os.path.join(RES, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=1, default=float)
    print('done')


if __name__ == '__main__':
    main()
