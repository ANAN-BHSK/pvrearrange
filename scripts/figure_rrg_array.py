"""Example of a reconfigured array, a test instance with a clustered shadow and three
irradiance levels: by default cluster_9x9_f0.4_L3 (group S2); `python scripts/figure_rrg_array.py 6`
draws the 6x6 instance cluster_6x6_f0.25_L3 (group S1) instead. (a) fixed TCT connection, (b) connection found by RRG
(variant B, tuned parameters, seed 0), (c) P-V curves of TCT, BRA (variant A) and RRG.
All values are computed here with the reference electrical model."""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pvrearrange import random_instance, exact, heuristics as H
from pvrearrange.regroup import rrg
from pvrearrange.power import TCTModel

OUT = 'results/figures'; os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, SURF = '#0b0b0b', '#52514e', '#8a8984', '#ffffff'
BUS = '#52514e'
SUN = LinearSegmentedColormap.from_list('sun', ['#4a3b22', '#a77a1e', '#f2c14e', '#fff1c4'])
VMIN, VMAX = 150, 1000
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'figure.facecolor': SURF, 'savefig.facecolor': SURF})

SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 9
SPEC = {6: dict(shaded_frac=0.25, seed=3006), 9: dict(shaded_frac=0.4, seed=3109)}[SIZE]   # as in scripts/run_full.py
inst = random_instance(SIZE, SIZE, kind='cluster', levels=3, **SPEC)
m, n = inst.m, inst.n
model = TCTModel()
R_tct = inst.identity()
R_bra = H.block_ra(inst, rule='random', then_ra=False, iters=10_000, seed=0)
R_rrg = rrg(inst, 10_000, seed=0, rule='random', block_iters=max(3, round(1.096 * n)))
opt = exact.solve_cpsat(inst, 'B', 'var', time_limit=60)
assert opt['status'] == 'OPTIMAL'
assert abs(inst.row_sums(R_rrg).std() - inst.row_sums(opt['R']).std()) < 1e-9, 'RRG not optimal here'
# within each electrical row, list modules by physical column for readability
R_rrg = np.array([sorted(r, key=lambda p: (p % n, p // n)) for r in R_rrg])


def label(p):            # module at physical row k, column j -> "kj" (1-based)
    return f'{p // n + 1}{p % n + 1}'


def array_panel(ax, R, x0, title, note):
    k = 6 / n; cw, ch, gap = 0.92 * k, 0.80 * k, 0.30 * k   # module width/height, vertical gap for the bus
    S = inst.row_sums(R)
    ax.text(x0 + n * cw / 2, 0.55, title, ha='center', va='bottom', fontsize=9, fontweight='bold', color=INK)
    ax.text(x0 + n * cw + 0.15, 0.25, 'S (W/m²)', ha='left', va='bottom', fontsize=7.2, color=INK2)
    for i in range(m):
        y = -i * (ch + gap)
        ax.text(x0 - 0.18, y - ch / 2, f'r{i + 1}', ha='right', va='center', fontsize=8.5 if n <= 6 else 7.5, style='italic', color=INK)
        # TCT ties: the node below row i is the node above row i+1
        if i == 0:
            ax.plot([x0 - 0.05, x0 + n * cw - 0.05], [y + gap / 2] * 2, color=BUS, lw=1.5, solid_capstyle='butt')
        ax.plot([x0 - 0.05, x0 + n * cw - 0.05], [y - ch - gap / 2] * 2, color=BUS, lw=1.5, solid_capstyle='butt')
        for j in range(n):
            p = R[i, j]; g = inst.flat[p]; c = SUN((g - VMIN) / (VMAX - VMIN))
            xl = x0 + j * cw
            ax.plot([xl + cw / 2 - .05] * 2, [y + gap / 2, y], color=BUS, lw=1.1)
            ax.plot([xl + cw / 2 - .05], [y + gap / 2], 'o', color=BUS, ms=2.6 * min(1, k * 1.2))
            ax.plot([xl + cw / 2 - .05] * 2, [y - ch, y - ch - gap / 2], color=BUS, lw=1.1)
            ax.plot([xl + cw / 2 - .05], [y - ch - gap / 2], 'o', color=BUS, ms=2.6 * min(1, k * 1.2))
            ax.add_patch(Rectangle((xl + 0.04, y - ch), cw - 0.18, ch, fc=c, ec=INK, lw=0.9))
            lum = 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2]
            tc = INK if lum > 0.55 else SURF
            if n <= 6:
                ax.text(xl + cw / 2 - .05, y - ch * 0.36, label(p), ha='center', va='center', fontsize=8.2, color=tc, fontweight='bold')
                ax.text(xl + cw / 2 - .05, y - ch * 0.74, f'{g:.0f}', ha='center', va='center', fontsize=6.3, color=tc)
            else:
                ax.text(xl + cw / 2 - .05, y - ch / 2, label(p), ha='center', va='center', fontsize=6.6, color=tc, fontweight='bold')
        ax.text(x0 + n * cw + 0.15, y - ch / 2, f'{S[i]:.0f}', ha='left', va='center', fontsize=8 if n <= 6 else 7.2, color=INK)
    ax.text(x0 + n * cw / 2, -m * (ch + gap) - 0.05, note, ha='center', va='top', fontsize=7.6, color=INK2, linespacing=1.4)


def curve(R):
    I, V, P = model.pv_curve([inst.flat[r].tolist() for r in R])
    o = np.argsort(V)
    return V[o], P[o]


res = {}
for key, R in (('TCT', R_tct), ('BRA', R_bra), ('RRG', R_rrg)):
    g = model.gmpp([inst.flat[r].tolist() for r in R]); res[key] = (g, inst.row_sums(R).std())

fig = plt.figure(figsize=(7.2, 7.4))
ax = fig.add_axes([0.0, 0.33, 1.0, 0.67]); ax.axis('off')
array_panel(ax, R_tct, 0.55, '(a) Fixed TCT connection',
            f'σ(S) = {res["TCT"][1]:.0f} W/m²   P = {res["TCT"][0]["P"]:.0f} W   {res["TCT"][0]["peaks"]} peak' + ('s' if res["TCT"][0]["peaks"] > 1 else ''))
array_panel(ax, R_rrg, 7.75, '(b) Connection found by RRG (variant B)',
            f'σ(S) = {res["RRG"][1]:.0f} W/m²   P = {res["RRG"][0]["P"]:.0f} W   {res["RRG"][0]["peaks"]} peak' + ('s' if res["RRG"][0]["peaks"] > 1 else ''))
ax.set_xlim(0, 14.7); ax.set_ylim(-6 * 1.10 - 0.75, 1.0)
ax.annotate('', xy=(7.25, -2.9), xytext=(6.75, -2.9), arrowprops=dict(arrowstyle='-|>', color=INK2, lw=1.4))
if n > 6:                                     # irradiance key instead of numbers in the cells
    lv = sorted(set(inst.flat.tolist()))
    x = 7.35 - 0.55 * len(lv)
    ax.text(x - 0.1, -7.2, 'W/m²:', ha='right', va='center', fontsize=7.2, color=INK2)
    for v in lv:
        c = SUN((v - VMIN) / (VMAX - VMIN))
        ax.add_patch(Rectangle((x, -7.32), 0.24, 0.24, fc=c, ec=INK, lw=0.6))
        ax.text(x + 0.3, -7.2, f'{v:.0f}', ha='left', va='center', fontsize=7.2, color=INK2); x += 1.1

bx = fig.add_axes([0.10, 0.085, 0.86, 0.21])
names = {'TCT': 'TCT (no reconfiguration)', 'BRA': 'BRA, variant A', 'RRG': 'RRG, variant B'}
cols = {'TCT': '#8a8984', 'BRA': '#eb6834', 'RRG': '#2a78d6'}
for key, R in (('TCT', R_tct), ('RRG', R_rrg), ('BRA', R_bra)):
    V, P = curve(R); g = res[key][0]
    bx.plot(V, P, color=cols[key], lw=1.8 if key != 'BRA' else 1.4, ls='-' if key != 'BRA' else '--',
            label=f'{names[key]}: {g["P"]:.0f} W')
    bx.plot([g['V']], [g['P']], 'o', color=cols[key], ms=5, mec=SURF, mew=0.8)
bx.set_xlim(0, None); bx.set_ylim(0, None)
bx.set_xlabel('array voltage (V)'); bx.set_ylabel('array power (W)')
bx.set_title('(c) P–V curves (dots: global maximum power point)', loc='left', fontsize=9, fontweight='bold')
for s in ('top', 'right'): bx.spines[s].set_visible(False)
bx.grid(color='#e4e3df', lw=0.6); h, l = bx.get_legend_handles_labels(); o = [0, 2, 1]
bx.legend([h[k] for k in o], [l[k] for k in o], frameon=False, fontsize=7.6, loc='upper left')
fig.text(0.5, 0.002, 'Labels kj: module in physical row k and column j. Modules stay in place; the switches connect each to the row bus shown.',
         ha='center', va='bottom', fontsize=7.0, color=MUTED)
fig.savefig(f'{OUT}/figR_rrg_array' + ('' if SIZE == 9 else f'_{SIZE}x{SIZE}') + '.png', dpi=300); plt.close(fig)
for k, (g, s) in res.items():
    print(k, f'sd={s:.1f}', f"P={g['P']:.1f}", f"peaks={g['peaks']}", f"V={g['V']:.1f}")
print('gain RRG vs TCT %.1f%%, vs BRA %.2f%%' % ((res['RRG'][0]['P'] / res['TCT'][0]['P'] - 1) * 100,
                                                (res['RRG'][0]['P'] / res['BRA'][0]['P'] - 1) * 100))
print(R_rrg)
