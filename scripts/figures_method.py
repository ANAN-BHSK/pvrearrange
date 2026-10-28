"""Explanatory figures for the manuscript: study overview, problem mapping with a worked
example, RRG worked example (Proposition 3), and RRG flowchart.
All numbers in the worked examples are computed here (CP-SAT optima, exchange optimality check)."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, '.')
from pvrearrange.instance import Instance
from pvrearrange import exact

OUT = 'results/figures'; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
BLUE, ORANGE, GREEN, AMBER = '#2a78d6', '#eb6834', '#1baf7a', '#eda100'
BOXF, BOXE = '#f3f2ee', '#b9b8b2'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})
SUN = LinearSegmentedColormap.from_list('sun', ['#4a3b22', '#a77a1e', '#f2c14e', '#fff1c4'])   # one hue, dark = shaded
VMIN, VMAX = 150, 1000


def matrix(ax, M, x0, y0, title=None, cell=0.55, sums=True, hl_cols=(), hl_color=BLUE, fmt=lambda v: f'{v:.0f}',
           sum_label='S', note=None):
    """Draw matrix M with its row totals; top-left corner at (x0, y0) in data units."""
    m, n = M.shape
    for i in range(m):
        for j in range(n):
            v = M[i, j]; c = SUN((v - VMIN) / (VMAX - VMIN))
            ax.add_patch(Rectangle((x0 + j * cell, y0 - (i + 1) * cell), cell, cell, fc=c, ec=SURF, lw=1.5))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            ax.text(x0 + (j + .5) * cell, y0 - (i + .5) * cell, fmt(v), ha='center', va='center', fontsize=7.2,
                    color=INK if lum > 0.55 else SURF)
    for j in hl_cols:
        ax.add_patch(Rectangle((x0 + j * cell + .03, y0 - m * cell + .03), cell - .06, m * cell - .06, fc='none',
                               ec=hl_color, lw=2.0))
    if sums:
        S = M.sum(axis=1)
        for i in range(m):
            ax.text(x0 + n * cell + .12, y0 - (i + .5) * cell, f'{S[i]:.0f}', ha='left', va='center', fontsize=7.6,
                    color=INK, fontweight='bold')
        ax.text(x0 + n * cell + .12, y0 + .14, sum_label, ha='left', va='bottom', fontsize=7.6, color=INK2, style='italic')
    if title:
        ax.text(x0, y0 + .42, title, ha='left', va='bottom', fontsize=8.6, color=INK, fontweight='bold')
    if note:
        ax.text(x0, y0 - m * cell - .16, note, ha='left', va='top', fontsize=7.4, color=INK2, linespacing=1.35)


def arrow(ax, p, q, text=None, color=INK2, rad=0.0, tpos=0.5, toff=(0, .14), fs=7.4, ha='center'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='-|>', mutation_scale=10, lw=1.2, color=color,
                                 connectionstyle=f'arc3,rad={rad}', shrinkA=2, shrinkB=2))
    if text:
        x = p[0] + (q[0] - p[0]) * tpos + toff[0]; y = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(x, y, text, ha=ha, va='bottom', fontsize=fs, color=INK2, linespacing=1.3)


def box(ax, x, y, w, h, text, fc=BOXF, ec=BOXE, fs=7.6, bold_first=True, lw=1.0, color=INK, ha='center', shape='round'):
    style = 'round,pad=0.02,rounding_size=0.08' if shape == 'round' else 'square,pad=0.02'
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fc, ec=ec, lw=lw))
    lines = text.split('\n')
    if bold_first and len(lines) > 1:
        ax.text(x + (w / 2 if ha == 'center' else .1), y + h - .13, lines[0], ha=ha, va='top', fontsize=fs + .3,
                color=color, fontweight='bold')
        ax.text(x + (w / 2 if ha == 'center' else .1), y + h - .40, '\n'.join(lines[1:]), ha=ha, va='top', fontsize=fs,
                color=color, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, color=color,
                fontweight='bold' if bold_first else 'normal', linespacing=1.3)


# ---------------------------------------------------------------- Fig: overview of the study
def fig_overview():
    import json
    D = json.load(open('results/analysis/manuscript_data.json'))
    hitA = {o['label']: o['hit_pct'] for o in D['methods']['A']}; hitB = {o['label']: o['hit_pct'] for o in D['methods']['B']}
    fig, ax = plt.subplots(figsize=(7.2, 3.3)); ax.set_xlim(0, 14.4); ax.set_ylim(0, 6.6); ax.axis('off')
    # column headers
    heads = [(1.45, 'Input'), (4.52, 'Problem variant'), (7.55, 'Equivalent problem'), (10.4, 'Method'), (13.1, 'Result')]
    for x, t in heads:
        ax.text(x, 6.42, t, ha='center', va='center', fontsize=8.2, fontweight='bold', color=INK2)
    # input: shaded array icon
    ax.add_patch(FancyBboxPatch((0.15, 1.95), 2.6, 3.9, boxstyle='round,pad=0.02,rounding_size=0.1', fc=BOXF, ec=BOXE, lw=1.0))
    ax.text(1.45, 5.55, 'Shaded TCT array', ha='center', va='center', fontsize=7.8, fontweight='bold', color=INK)
    G = np.array([[200, 500, 800, 1000], [200, 500, 1000, 1000], [500, 800, 1000, 1000], [1000] * 4], float)
    cs = 0.42; gx, gy = 0.6, 5.15
    for i in range(4):
        for j in range(4):
            ax.add_patch(Rectangle((gx + j * cs, gy - (i + 1) * cs), cs, cs, fc=SUN((G[i, j] - VMIN) / (VMAX - VMIN)), ec=SURF, lw=1.2))
    ax.text(1.45, 3.15, 'Choose which modules\nform each electrical row\nso that row totals are\nas equal as possible', ha='center',
            va='top', fontsize=6.9, color=INK2, linespacing=1.3)
    W, Hh = 2.6, 1.25
    lanes = [  # y, variant, equivalent, method, result, highlight
        (4.15, 'Variant A\nmodules stay in\ntheir column', 'Matrix rearrangement\npermute entries\nwithin columns',
         'RA and BRA\nrearrangement\nheuristics', f'Best-known solution\nin {hitA["BRA"]:.0f}% of runs\n(BRA)', False),
        (2.2, 'Variant B\nany module in\nany row', 'Balanced partitioning\nn modules per row\n(Section 4)',
         'RRG (proposed)\nvirtual columns\n+ regrouping', f'Best-known solution\nin {hitB["RRG"]:.0f}% of runs\n(RRG)', True),
    ]
    xs = [3.3, 6.2, 9.25, 11.9]; ws = [2.45, 2.7, 2.3, 2.4]
    for y, *texts, hl in lanes:
        for k, (x, w, t) in enumerate(zip(xs, ws, texts)):
            h = hl and k >= 2
            box(ax, x, y, w, Hh, t, fc='#e8f0fb' if h else BOXF, ec=BLUE if h else BOXE, lw=1.6 if h else 1.0, fs=6.9)
        for k in range(3):
            arrow(ax, (xs[k] + ws[k] + .02, y + Hh / 2), (xs[k + 1] - .02, y + Hh / 2))
        arrow(ax, (2.77, y + Hh / 2), (xs[0] - .02, y + Hh / 2))
    # common evaluation band
    ax.add_patch(FancyBboxPatch((3.3, 0.08), 11.0, 1.42, boxstyle='round,pad=0.02,rounding_size=0.1', fc=SURF, ec=MUTED, lw=1.0, ls='--'))
    ax.text(3.45, 1.4, 'Common evaluation', fontsize=7.4, fontweight='bold', color=INK, va='top')
    ax.text(3.45, 1.0, f'{D["summary"]["n_instances"]} instances from 6×6 to 30×30 modules · proven optima or bounds from CP-SAT ·\n'
            'five metaheuristics as baselines · parameters tuned on separate instances ·\n'
            'simulated array power · four alternative electrical models on 38 small instances', fontsize=6.8, color=INK2, va='top', linespacing=1.35)
    fig.subplots_adjust(0.005, 0.005, 0.995, 0.995); fig.savefig(f'{OUT}/figM1_overview.png', dpi=300); plt.close(fig)


# ---------------------------------------------------------------- Fig: problem mapping, worked example
def fig_problem():
    G = np.array([[200, 500, 800, 1000], [200, 500, 1000, 1000], [500, 800, 1000, 1000], [1000] * 4], float)
    inst = Instance('example', G)
    out = {}
    for v in 'AB':
        r = exact.solve_cpsat(inst, v, 'var', workers=2)
        assert r['status'] == 'OPTIMAL'
        out[v] = inst.flat[r['R']]
    A = out['A']; Bm = out['B']
    # keep variant-A entries in their physical column (CP-SAT returns modules per column already)
    sd = lambda M: M.sum(axis=1).std()
    fig, ax = plt.subplots(figsize=(7.2, 2.5)); ax.set_xlim(0, 14.4); ax.set_ylim(0.15, 4.95); ax.axis('off')
    c = 0.6
    # panel (a): array with shadow -> fixed TCT connection
    x0 = 0.25
    matrix(ax, G, x0, 4.3, '(a) Shaded array, fixed TCT', cell=c,
           note=f'rows: electrical rows in series\nσ(S) = {sd(G):.0f} W/m²')
    x1 = 5.05
    matrix(ax, A, x1, 4.3, '(b) Variant A optimum', cell=c,
           note=f'entries permuted within columns\nσ(S) = {sd(A):.0f} W/m²')
    x2 = 9.85
    matrix(ax, Bm, x2, 4.3, '(c) Variant B optimum', cell=c,
           note=f'any module in any row, 4 per row\nσ(S) = {sd(Bm):.0f} W/m²')
    arrow(ax, (x0 + 4 * c + .95, 3.1), (x1 - .15, 3.1))
    arrow(ax, (x1 + 4 * c + .95, 3.1), (x2 - .15, 3.1))
    ax.text(7.2, 0.2, 'Cell values: module irradiance (W/m²); S: row totals. Optima computed with the exact model.',
            ha='center', va='bottom', fontsize=7.2, color=MUTED)
    fig.tight_layout(pad=0.2); fig.savefig(f'{OUT}/figM2_problem.png', dpi=300); plt.close(fig)
    return dict(G=G, A=A, B=Bm, sd=(sd(G), sd(A), sd(Bm)))


# ---------------------------------------------------------------- Fig: RRG worked example
def exchange_optimal(M):
    S = M.sum(1); m, n = M.shape
    for a in range(m):
        for b in range(a + 1, m):
            for i in range(n):
                for j in range(n):
                    if (M[a, i] - M[b, j]) * ((S[a] - M[a, i]) - (S[b] - M[b, j])) > 1e-9: return False
    return True


def ra_changes(M):
    """Would RA change any column of this grouping? (opposite-order violation)"""
    S = M.sum(1)
    for j in range(M.shape[1]):
        c = M[:, j]; r = S - c
        if any((c[a] - c[b]) * (r[a] - r[b]) > 1e-9 for a in range(len(c)) for b in range(len(c))): return True
    return False


def block_step(M, blk):
    rest = [j for j in range(M.shape[1]) if j not in blk]
    P = M[:, blk].sum(1); Q = M[:, rest].sum(1)
    oP = np.argsort(-P, kind='stable'); oQ = np.argsort(Q, kind='stable')
    N = M.copy(); Bk = M[:, blk]; nb = np.zeros_like(Bk); nb[oQ] = Bk[oP]; N[:, blk] = nb
    return N


def fig_rrg_example():
    M = np.array([[500, 400, 500, 400], [400, 300, 500, 500], [600, 200, 600, 200]], float)
    assert exchange_optimal(M)
    Mg = np.array([np.sort(r)[::-1] for r in M])            # sorted regrouping
    assert np.allclose(Mg.sum(1), M.sum(1)) and not ra_changes(Mg)
    blk = [0, 2]
    M2 = block_step(Mg, blk)
    assert M2.sum(1).var() < M.sum(1).var()
    v = lambda X: X.sum(1).var()
    fig, ax = plt.subplots(figsize=(7.2, 2.25)); ax.set_xlim(0, 14.4); ax.set_ylim(0.75, 4.75); ax.axis('off')
    c = 0.62
    xa, xb, xc = 0.2, 5.0, 9.8
    matrix(ax, M, xa, 4.0, '(a) Exchange optimum', cell=c,
           note=f'no exchange of two modules\nimproves; RA changes no column\nVar(S) = {v(M):.0f}')
    matrix(ax, Mg, xb, 4.0, '(b) Regrouped', cell=c, hl_cols=blk,
           note=f'entries permuted within rows:\nrow totals unchanged\nblock = columns 1 and 3 (outlined)')
    matrix(ax, M2, xc, 4.0, '(c) After one block step', cell=c, hl_cols=blk, hl_color=GREEN,
           note=f'block arranged oppositely to the\nother columns: rows 1 and 3 exchange\ntwo modules at once; Var(S) = {v(M2):.0f}')
    arrow(ax, (xa + 4 * c + .85, 3.05), (xb - .12, 3.05), 'regroup', toff=(0, .1))
    arrow(ax, (xb + 4 * c + .85, 3.05), (xc - .12, 3.05), 'block step', toff=(0, .1))
    fig.tight_layout(pad=0.2); fig.savefig(f'{OUT}/figM3_rrg_example.png', dpi=300); plt.close(fig)
    return dict(M=M, Mg=Mg, M2=M2, v=(v(M), v(M2)))


# ---------------------------------------------------------------- Fig: RRG flowchart
def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); ax.set_xlim(0, 14.4); ax.set_ylim(0, 8.8); ax.axis('off')
    W, H = 5.8, 0.8; cx = 3.65; x = cx - W / 2

    def node(y, text, hl=False, h=H, bold=False):
        box(ax, x, y, W, h, text, fc='#e8f0fb' if hl else BOXF, ec=BLUE if hl else BOXE, lw=1.5 if hl else 1.0,
            fs=7.3, bold_first=bold)

    def diamond(cy, text, w=3.9, h=1.05):
        ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)], closed=True,
                             fc='#fff6df', ec=AMBER, lw=1.2))
        ax.text(cx, cy, text, ha='center', va='center', fontsize=7.3, color=INK)

    yS, yR, d1, yG, yR2, d2, yA = 7.85, 6.7, 5.6, 3.95, 2.8, 1.68, 0.2
    node(yS, 'Start: R ← BLDM solution\n(rows = electrical rows; columns = virtual columns)')
    node(yR, 'Rearrange(R);   R* ← R', hl=True)
    diamond(d1, 'evaluations used < E ?')
    node(yG, 'Regroup R*: permute the entries within every row\n(row totals and objective unchanged)', hl=True)
    node(yR2, "R′ ← Rearrange(regrouped R*)", hl=True)
    diamond(d2, "Var(S(R′)) ≤ Var(S(R*)) ?", w=4.3)
    node(yA, "R* ← R′", h=0.68)
    for a, b in ((yS, yR + H), (yR, d1 + .53), (d1 - .53, yG + H), (yG, yR2 + H), (yR2, d2 + .53), (d2 - .53, yA + .68)):
        arrow(ax, (cx, a), (cx, b))
    ax.text(cx + .2, (d1 - .53 + yG + H) / 2, 'yes', fontsize=7.0, color=INK2, va='center')
    ax.text(cx + .2, (d2 - .53 + yA + .68) / 2, 'yes', fontsize=7.0, color=INK2, va='center')
    # exit
    ax.add_patch(FancyArrowPatch((cx + 1.95, d1), (cx + 3.35, d1), arrowstyle='-|>', mutation_scale=10, lw=1.2, color=INK2))
    ax.text(cx + 2.05, d1 + .08, 'no', fontsize=7.0, color=INK2, va='bottom')
    box(ax, cx + 3.4, d1 - .38, 1.45, .76, 'return R*', fs=7.4, bold_first=True)
    # loops back to the first decision on the left side
    lx = x - .35
    ax.plot([cx - 2.15, lx], [d2, d2], color=INK2, lw=1.2)
    ax.text(cx - 2.2, d2 + .08, 'no', fontsize=7.0, color=INK2, ha='right', va='bottom')
    ax.plot([x, lx, lx], [yA + .34, yA + .34, d1], color=INK2, lw=1.2)
    ax.add_patch(FancyArrowPatch((lx, d1), (cx - 1.95, d1), arrowstyle='-|>', mutation_scale=10, lw=1.2, color=INK2))
    # Rearrange detail panel
    bx, by, bw, bh = 9.35, 1.7, 4.95, 5.55
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.02,rounding_size=0.1', fc=SURF, ec=BLUE, lw=1.2, ls='--'))
    ax.text(bx + .2, by + bh - .18, 'Rearrange(R)', fontsize=8.2, fontweight='bold', color=BLUE, va='top')
    txt = ['1. RA: arrange each column oppositely\n    to the sum of the other columns;\n    sweep until no column changes',
           '2. Block steps: a random group of\n    columns is arranged oppositely to\n    the rest and kept if Var(S)\n    decreases; stop after q consecutive\n    failures',
           '3. RA again until no column changes']
    yy = by + bh - .75
    for t in txt:
        ax.text(bx + .2, yy, t, fontsize=7.1, color=INK, va='top', linespacing=1.35)
        yy -= 0.33 * (t.count('\n') + 1) + .35
    ax.text(bx + .2, by + .2, 'Each column or block step counts as\none evaluation; no step increases Var(S).',
            fontsize=6.8, color=INK2, va='bottom', linespacing=1.35)
    ax.plot([x + W, bx], [yR + H / 2, by + bh - .3], color=BLUE, lw=0.9, ls=':')
    ax.plot([x + W, bx], [yR2 + H / 2, by + .9], color=BLUE, lw=0.9, ls=':')
    fig.tight_layout(pad=0.2); fig.savefig(f'{OUT}/figM4_rrg_flowchart.png', dpi=300); plt.close(fig)


if __name__ == '__main__':
    fig_overview()
    p = fig_problem(); print('problem sd', [round(x, 1) for x in p['sd']]); print(p['A']); print(p['B'])
    e = fig_rrg_example(); print('rrg example var', e['v']); print(e['Mg']); print(e['M2'])
    fig_flowchart()
    print('written')
