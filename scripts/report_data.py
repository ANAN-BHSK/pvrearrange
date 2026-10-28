"""Collect every number used in the results report into results/analysis/report_data.json."""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root
import json
import numpy as np
import pandas as pd

A = 'results/analysis'
S = json.load(open(f'{A}/summary.json'))
ok = pd.read_csv(f'{A}/all_runs_with_gap.csv.gz'); ok['levels'] = ok['levels'].astype(str)
t2 = pd.read_csv(f'{A}/rq2_methods.csv')
D = {'summary': S}

ORDER_A = ['TCT', 'Greedy', 'RA', 'BRA', 'BRAVE+RA', 'ILS', 'RandomSearch', 'RKGA', 'PSO', 'DE', 'GOA', 'HHO']
ORDER_B = ['BLDM', 'BWSA', 'ILS', 'BLDM+ILS', 'RRG', 'RandomSearch', 'RKGA', 'PSO', 'DE', 'GOA', 'HHO']
D['methods'] = {}
for v, order in (('A', ORDER_A), ('B', ORDER_B)):
    t = t2[t2.variant == v].set_index('label')
    D['methods'][v] = [dict(label=l, **{k: float(t.at[l, k]) for k in ['median_gap', 'q90_gap', 'hit_pct', 'median_time_s',
                                                                         'below_tct_power_pct', 'median_power_gain_pct', 'runs']})
                       for l in order if l in t.index]
cp = __import__('pvrearrange.resultsio', fromlist=['read_runs']).read_runs()
cp = cp[cp.method == 'CP-SAT']; cp['levels'] = cp['levels'].astype(str); cp['optimal'] = cp.status == 'OPTIMAL'
s1 = cp[cp.set == 'S1']
D['cpsat'] = []
for v in 'AB':
    for m in sorted(s1.m.unique()):
        row = dict(variant=v, m=int(m))
        for L in ['1', '3', 'continuous']:
            d = s1[(s1.variant == v) & (s1.m == m) & (s1.levels == L)]
            row[f'opt_{L}'] = float(d.optimal.mean() * 100) if len(d) else None
            dd = d[d.optimal]
            row[f't_{L}'] = float(dd.time.median()) if len(dd) else None
        D['cpsat'].append(row)
import os
if os.path.exists(f'{A}/rq1_cpsat_combined.csv'):
    c2 = pd.read_csv(f'{A}/rq1_cpsat_combined.csv'); c2['levels'] = c2['levels'].astype(str)
    for row in D['cpsat']:
        d = c2[(c2.set == 'S1') & (c2.variant == row['variant']) & (c2.m == row['m'])]
        for L in ['1', '3', 'continuous']:
            dl = d[d.levels == L]
            row[f'opt2_{L}'] = float(dl.optimal2.mean() * 100) if len(dl) else None
            row[f't2_{L}'] = float(dl[dl.optimal2].time2.median()) if dl.optimal2.any() else None
    D['cpsat_combined_by_set'] = {f'{v}|{s}': [int(g.optimal2.sum()), int(len(g))] for (v, s), g in c2.groupby(['variant', 'set'])}
    D['cpsat_combined_by_levels'] = {f'{v}|{L}': [int(g.optimal2.sum()), int(len(g))] for (v, L), g in c2.groupby(['variant', 'levels'])}
    tp = pd.read_csv(f'{A}/rq2_methods_proven.csv')
    D['methods_proven'] = {v: tp[tp.variant == v].set_index('label').drop(columns='variant').to_dict('index') for v in 'AB'}
cpo = ok[ok.method == 'CP-SAT']
D['cpsat_incumbent'] = {v: dict(median_gap_not_opt=float(cpo[(cpo.variant == v) & (cpo.status != 'OPTIMAL')].gap.median()),
                                below_tct_power_pct=float(cpo[cpo.variant == v].below_tct.mean() * 100),
                                n_not_opt=int((cpo[(cpo.variant == v)].status != 'OPTIMAL').sum()))
                        for v in 'AB'}
gl = pd.read_csv(f'{A}/rq2_gap_by_levels.csv'); hl = pd.read_csv(f'{A}/rq2_hit_by_levels.csv')
D['by_levels'] = {'gap': gl.to_dict('records'), 'hit': hl.to_dict('records')}
D['by_size'] = pd.read_csv(f'{A}/rq2_gap_by_size.csv').to_dict('records')
D['time_by_size'] = pd.read_csv(f'{A}/rq2_time_by_size.csv').to_dict('records')
D['a12'] = pd.read_csv(f'{A}/rq2_a12.csv').to_dict('records')
D['tuned_vs_default'] = pd.read_csv(f'{A}/tuned_vs_default.csv').to_dict('records')
D['anytime'] = pd.read_csv(f'{A}/anytime_gap.csv').to_dict('records')

# shaded fraction and kind effects (S1+S2, sizes 9 and 16, levels 3/cont)
d = ok[ok.set.isin(['S1', 'S2']) & ok.m.isin([9, 16]) & ok.levels.isin(['3', 'continuous'])
       & ok.label.isin(['RA', 'BRA', 'ILS', 'RKGA', 'RandomSearch', 'BLDM'])]
D['by_frac'] = d.pivot_table(index=['variant', 'label'], columns='frac', values='gap', aggfunc='median').round(4).reset_index().to_dict('records')
D['by_kind'] = ok[ok.set == 'S1'].pivot_table(index=['variant', 'label'], columns='kind', values='gap', aggfunc='median').round(4).reset_index().to_dict('records')
D['ageing'] = ok[ok.set == 'S3'].pivot_table(index=['variant', 'label'], columns='sigma', values='gap', aggfunc='median').round(4).reset_index().to_dict('records')
D['standard'] = ok[ok.set == 'S4'].groupby(['variant', 'label']).agg(median_gap=('gap', 'median'), hit=('hit', 'mean')).round(4).reset_index().to_dict('records')

# power summary
tp = ok[ok.method == 'TCT'].set_index('instance').power
bp = ok.groupby('instance').power.max()
D['power'] = dict(median_gain_best=float(((bp / tp.reindex(bp.index)) - 1).median() * 100),
                  q10=float(((bp / tp.reindex(bp.index)) - 1).quantile(0.1) * 100),
                  q90=float(((bp / tp.reindex(bp.index)) - 1).quantile(0.9) * 100))

# sensitivity + RQ3 (small instances)
p = pd.read_csv('results/sensitivity_power.csv'); c = pd.read_csv('results/rq3_spearman.csv')
w = p.pivot_table(index=['instance', 'model'], columns='config', values='power_W').reset_index()
sur = ['opt_var', 'opt_L1', 'opt_range', 'opt_bot']
for o in sur: w[o + '_gain'] = (w[o] / w.TCT - 1) * 100
w['best'] = w[sur].idxmax(axis=1)
w['spread'] = (w[sur].max(axis=1) / w[sur].min(axis=1) - 1) * 100
D['sensitivity'] = []
base = w[w.model == 'base'].set_index('instance')
for mdl, g in w.groupby('model'):
    gi = g.set_index('instance')
    D['sensitivity'].append(dict(model=mdl, **{o: float(g[o + '_gain'].median()) for o in sur},
                                 below_tct=float((g[[o + '_gain' for o in sur]] < -0.01).any(axis=1).mean() * 100),
                                 var_best_pct=float((g.best == 'opt_var').mean() * 100),
                                 spread_max=float(g.spread.max()),
                                 agree_base=float((gi.best == base.best.reindex(gi.index)).mean() * 100)))
D['sensitivity_n'] = int(p.instance.nunique())
D['sensitivity_not_optimal'] = int((p[p.model == 'base'].status == 'FEASIBLE').sum())
D['rq3_spearman'] = c.groupby('objective').spearman.agg(['min', 'median', 'mean']).round(3).reset_index().to_dict('records')
c1 = c.drop_duplicates('instance').set_index('instance')
bs = base[sur].max(axis=1)
D['rq3_regret'] = dict(best_surrogate_max=float(((1 - bs.reindex(c1.index) / c1.p_best) * 100).max()),
                       ls_from_tct_median=float(((1 - c1.p_ls_from_tct / c1.p_best) * 100).median()),
                       ls_from_tct_max=float(((1 - c1.p_ls_from_tct / c1.p_best) * 100).max()),
                       ls_from_tct_pct_worse=float((((1 - c1.p_ls_from_tct / c1.p_best) * 100) > 0.01).mean() * 100))
D['tuning'] = json.load(open('results/tuning_history.json'))
D['tuned'] = json.load(open('results/tuned_params.json'))
def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)): return None if np.isnan(x) else float(x)
    if isinstance(x, np.integer): return int(x)
    return x
json.dump(clean(D), open(f'{A}/report_data.json', 'w'), indent=1, default=float, allow_nan=False)
print('ok', len(json.dumps(D)))
