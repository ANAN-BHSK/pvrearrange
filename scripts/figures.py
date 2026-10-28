"""Figures for the results report (static, print-oriented, light surface).
Palette: reference categorical slots 1-4 (validated: adjacent CVD dE >= 9.1); slots 3-4 are
below 3:1 contrast, so every line is directly labelled and carries a distinct marker."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = 'results/figures'; os.makedirs(OUT, exist_ok=True)
SER = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
INK, INK2, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff'
MARK = ['o', 's', '^', 'D']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.titlesize': 10, 'axes.titleweight': 'bold',
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': SURF,
                     'axes.facecolor': SURF, 'savefig.facecolor': SURF})


def _grid(ax):
    ax.grid(True, axis='y', color=GRID, lw=0.8); ax.set_axisbelow(True)


def _label_end(ax, x, y, text, color_idx, dy=0):
    ax.annotate(text, (x[-1], y[-1]), xytext=(6, dy), textcoords='offset points', va='center', fontsize=8, color=INK)


def fig_cpsat(df):
    cp = df[df.method == 'CP-SAT'].copy(); cp['opt'] = (cp.status == 'OPTIMAL') * 100
    cp = cp[cp.set == 'S1']
    comb = 'results/analysis/rq1_cpsat_combined.csv'
    c2 = None
    if os.path.exists(comb):
        c2 = pd.read_csv(comb); c2 = c2[c2.set == 'S1']; c2['levels'] = c2['levels'].astype(str); c2['opt'] = c2.optimal2 * 100
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    names = {'1': '1 shaded level', '3': '3 levels', 'continuous': 'continuous'}
    for ax, v in zip(axs, 'AB'):
        t = cp[cp.variant == v].pivot_table(index='m', columns='levels', values='opt', aggfunc='mean')
        t2 = None if c2 is None else c2[c2.variant == v].pivot_table(index='m', columns='levels', values='opt', aggfunc='mean')
        for k, L in enumerate(['1', '3', 'continuous']):
            if L not in t: continue
            off = (k - 1) * 0.35
            ax.plot(t.index + off, t[L], color=SER[k], lw=2, marker=MARK[k], ms=6, mec=SURF, mew=1, label=f'{names[L]}, 60 s')
            if t2 is not None and L in t2:
                ax.plot(t2.index + off, t2[L], color=SER[k], lw=1.4, ls='--', marker=MARK[k], ms=5, mfc=SURF, mec=SER[k],
                        mew=1.2, label=f'{names[L]}, 600 s')
        ax.set_title(f'Variant {v} ({"within column" if v == "A" else "free rows"})', loc='left')
        ax.set_xlabel('array size m (m × m)'); ax.set_xticks(sorted(cp.m.unique()))
        ax.set_ylim(-5, 108); ax.set_xlim(4, 32); _grid(ax)
    axs[0].set_ylabel('solved to proven optimality (%)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc='lower center', ncol=3, fontsize=7.5, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.13, 1, 1)); fig.savefig(f'{OUT}/fig1_cpsat.png', dpi=300); plt.close(fig)


def fig_profile(ok, picks, fname, title):
    """ECDF over instances of the per-instance median gap, one panel per variant."""
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, v in zip(axs, 'AB'):
        d = ok[(ok.variant == v) & (ok.params != 'default')]
        med = d.groupby(['instance', 'label']).gap.median().unstack()
        xs = np.logspace(-4, 1.5, 300)
        for k, lab in enumerate(picks[v]):
            if lab not in med: continue
            g = med[lab].values
            y = [(g <= x).mean() * 100 for x in xs]
            y0 = (g <= 1e-7).mean() * 100
            ax.plot(xs, y, color=SER[k], lw=2, marker=MARK[k], ms=5, mec=SURF, mew=1,
                    markevery=(k * 9, 37), label=lab.replace('RandomSearch', 'random search'))
        ax.set_xscale('log'); ax.set_xlim(1e-4, 10 ** 1.5)
        ax.set_title(f'Variant {v}', loc='left'); ax.set_xlabel('median gap, % of mean row total (log)')
        _grid(ax); ax.set_ylim(0, 102)
        ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=4, fontsize=7.5,
                  handlelength=1.6, columnspacing=1.0)
    axs[0].set_ylabel('instances with median gap ≤ x (%)')
    fig.tight_layout(); fig.savefig(f'{OUT}/{fname}', dpi=300); plt.close(fig)


def fig_anytime(an):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, (v, picks) in zip(axs, (('A', ['RA', 'RKGA', 'DE', 'RandomSearch']), ('B', ['RRG', 'RKGA', 'DE', 'RandomSearch']))):
        d = an[an.variant == v]
        for k, lab in enumerate(picks):
            s = d[d.label == lab].sort_values('evals')
            if s.empty: continue
            ax.plot(s.evals, s.median_gap + 1e-4, color=SER[k], lw=2, marker=MARK[k], ms=6, mec=SURF, mew=1, label=lab.replace('RandomSearch', 'random search'))
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(800, 12000)
        ax.legend(frameon=False, fontsize=8, loc='best')
        ax.set_title(f'Variant {v}', loc='left'); ax.set_xlabel('evaluations used (log)'); _grid(ax)
    axs[0].set_ylabel('median gap + 10⁻⁴, % (log)')
    fig.tight_layout(); fig.savefig(f'{OUT}/fig3_anytime.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    import glob
    ok = pd.read_csv('results/analysis/all_runs_with_gap.csv.gz')
    ok['levels'] = ok['levels'].astype(str)
    raw = __import__('pvrearrange.resultsio', fromlist=['read_runs']).read_runs()
    raw['levels'] = raw['levels'].astype(str)
    fig_cpsat(raw)
    t2 = pd.read_csv('results/analysis/rq2_methods.csv')
    mh = ['RKGA', 'PSO', 'DE', 'GOA', 'HHO']
    bestA = t2[(t2.variant == 'A') & t2.label.isin(mh)].sort_values('median_gap').label.iloc[0]
    bestB = t2[(t2.variant == 'B') & t2.label.isin(mh)].sort_values('median_gap').label.iloc[0]
    fig_profile(ok, {'A': ['BRA', 'RA', bestA, 'RandomSearch'], 'B': ['RRG', 'BLDM+ILS', bestB, 'ILS']},
                'fig2_gap_profile.png', 'Share of instances solved within a given gap')
    fig_anytime(pd.read_csv('results/analysis/anytime_long.csv'))
    print('figures written; best MH A/B:', bestA, bestB)
