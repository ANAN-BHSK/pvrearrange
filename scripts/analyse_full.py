"""Statistical analysis of the full test run (protocol Sec. 9-10).

gap  = (sd(S) - sd*) / mean row sum x 100, sd* = best known standard deviation of row sums
       for the instance and variant (all methods; for variant B also all variant-A
       solutions, which are feasible for B).
hit  = run reached the best known value.
Outputs tables to results/analysis/*.csv and a summary JSON.
"""
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # repository root
import glob, json, os, sys, warnings
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, spearmanr
warnings.filterwarnings('ignore')
OUT = 'results/analysis'; os.makedirs(OUT, exist_ok=True)

df = __import__('pvrearrange.resultsio', fromlist=['read_runs']).read_runs()
df['levels'] = df['levels'].astype(str)
df['label'] = np.where(df.params == 'default', df.method + ' (default)', df.method)
ok = df.dropna(subset=['var']).copy()
ok['sd'] = np.sqrt(ok['var'].clip(lower=0))
best = ok.groupby(['instance', 'variant'])['sd'].min().unstack()
# long CP-SAT re-run (600 s, 8 threads, warm start) on every case not proven optimal in 60 s
LONG = 'results/cpsat_long_all.csv'
lg = pd.read_csv(LONG) if os.path.exists(LONG) else None
if lg is not None:
    lsd = np.sqrt(lg.set_index(['instance', 'variant']).best_var.clip(lower=0)).unstack()
    best = np.minimum(best, lsd.reindex(index=best.index, columns=best.columns).fillna(np.inf))
best['B'] = np.minimum(best['B'], best['A'])
ok['sd_best'] = [best.at[i, v] for i, v in zip(ok.instance, ok.variant)]
ok['gap'] = (ok.sd - ok.sd_best) / ok.mean_row * 100
tol = 1e-9 * ok.mean_row
ok['hit'] = ok.sd <= ok.sd_best + tol
tct_p = ok[ok.method == 'TCT'].set_index('instance')['power']
ok['power_gain'] = (ok.power / ok.instance.map(tct_p) - 1) * 100
ok['below_tct'] = ok.power < ok.instance.map(tct_p) - 0.5
S = {}
S['n_instances'] = int(df.instance.nunique()); S['n_rows'] = int(len(df))
S['by_set'] = df.drop_duplicates('instance').set.value_counts().to_dict()

# ---- RQ1: exact method -----------------------------------------------------------
cp = df[df.method == 'CP-SAT'].copy(); cp['optimal'] = cp.status == 'OPTIMAL'
t1 = cp.pivot_table(index=['variant', 'm'], columns='levels', values='optimal', aggfunc='mean').mul(100).round(0)
t1t = cp[cp.optimal].pivot_table(index=['variant', 'm'], columns='levels', values='time', aggfunc='median').round(2)
t1.to_csv(f'{OUT}/rq1_cpsat_optimal_pct.csv'); t1t.to_csv(f'{OUT}/rq1_cpsat_time_median.csv')
cpg = ok[ok.method == 'CP-SAT']
S['cpsat_optimal_pct'] = {v: round(100 * cp[cp.variant == v].optimal.mean(), 1) for v in 'AB'}
S['cpsat_gap_when_not_optimal_median'] = round(float(cpg[cpg.status != 'OPTIMAL'].gap.median()), 4)
S['cpsat_no_solution'] = int((cp.status.isin(['UNKNOWN', 'MODEL_INVALID'])).sum())
S['best_known_proven_pct'] = {}
proven = {v: set(cp[(cp.variant == v) & cp.optimal].instance) for v in 'AB'}
if lg is not None:
    for v in 'AB': proven[v] |= set(lg[(lg.variant == v) & (lg.status == 'OPTIMAL')].instance)
for v in 'AB':
    S['best_known_proven_pct'][v] = round(100 * len(proven[v]) / S['n_instances'], 1)
if lg is not None:
    # combined status: proven in 60 s, or proven in the long run
    c2 = cp[['instance', 'variant', 'set', 'm', 'levels', 'optimal', 'time']].copy()
    lgi = lg.set_index(['instance', 'variant'])
    key = list(zip(c2.instance, c2.variant))
    c2['long_opt'] = [k in lgi.index and lgi.at[k, 'status'] == 'OPTIMAL' for k in key]
    c2['optimal2'] = c2.optimal | c2.long_opt
    c2['time2'] = np.where(c2.optimal, c2.time, [lgi.at[k, 'time_s'] if k in lgi.index else np.nan for k in key])
    t1b = c2.pivot_table(index=['variant', 'm'], columns='levels', values='optimal2', aggfunc='mean').mul(100).round(0)
    t1bt = c2[c2.optimal2].pivot_table(index=['variant', 'm'], columns='levels', values='time2', aggfunc='median').round(2)
    t1b.to_csv(f'{OUT}/rq1_cpsat_long_optimal_pct.csv'); t1bt.to_csv(f'{OUT}/rq1_cpsat_long_time_median.csv')
    lg2 = lg.copy()
    lg2['improved_hint'] = lg2.best_var < lg2.hint_var - 1e-6 * (1 + lg2.hint_var)
    lg2['bound_gap'] = (np.sqrt(lg2.best_var.clip(lower=0)) - np.sqrt(lg2.bound_var.clip(lower=0)))
    mr = ok.drop_duplicates('instance').set_index('instance').mean_row
    lg2['bound_gap_pct'] = lg2.bound_gap / lg2.instance.map(mr) * 100
    S['cpsat_long'] = dict(
        cases=int(len(lg)), optimal=int((lg.status == 'OPTIMAL').sum()),
        optimal_by_variant={v: [int(((lg.variant == v) & (lg.status == 'OPTIMAL')).sum()), int((lg.variant == v).sum())] for v in 'AB'},
        optimal_pct_total={v: round(100 * c2[c2.variant == v].optimal2.mean(), 1) for v in 'AB'},
        median_time_optimal=float(lg[lg.status == 'OPTIMAL'].time_s.median()),
        improved_hint=int(lg2.improved_hint.sum()),
        improved_hint_optimal=int((lg2.improved_hint & (lg2.status == 'OPTIMAL')).sum()),
        open_max_bound_gap_pct=float(lg2[lg2.status != 'OPTIMAL'].bound_gap_pct.max()),
        open_median_bound_gap_pct=float(lg2[lg2.status != 'OPTIMAL'].bound_gap_pct.median()),
        open_cases=lg2[lg2.status != 'OPTIMAL'][['instance', 'variant', 'bound_gap_pct']].round(4).to_dict('records'),
        threads=int(lg.threads.iloc[0]), time_limit=float(lg.time_limit_s.iloc[0]))
    c2.to_csv(f'{OUT}/rq1_cpsat_combined.csv', index=False)

# ---- RQ2: heuristic quality ------------------------------------------------------
stoch = ok[~ok.method.isin(['CP-SAT'])]
g = stoch.groupby(['variant', 'label'])
t2 = pd.DataFrame({'median_gap': g.gap.median(), 'q90_gap': g.gap.quantile(0.9), 'hit_pct': g.hit.mean() * 100,
                   'median_time_s': g.time.median(), 'below_tct_power_pct': g.below_tct.mean() * 100,
                   'median_power_gain_pct': g.power_gain.median(), 'runs': g.size()}).round(4)
t2.to_csv(f'{OUT}/rq2_methods.csv')
ok['proven'] = [i in proven[v] for i, v in zip(ok.instance, ok.variant)]
pr = ok[ok.proven & (ok.method != 'CP-SAT')].groupby(['variant', 'label'])
t2p = pd.DataFrame({'instances': pr.instance.nunique(), 'median_gap': pr.gap.median(), 'q90_gap': pr.gap.quantile(0.9),
                    'max_gap': pr.gap.max(), 'hit_pct': pr.hit.mean() * 100}).round(4)
t2p.to_csv(f'{OUT}/rq2_methods_proven.csv')
t2l = stoch.pivot_table(index=['variant', 'label'], columns='levels', values='gap', aggfunc='median').round(4)
t2l.to_csv(f'{OUT}/rq2_gap_by_levels.csv')
t2h = stoch.pivot_table(index=['variant', 'label'], columns='levels', values='hit', aggfunc='mean').mul(100).round(1)
t2h.to_csv(f'{OUT}/rq2_hit_by_levels.csv')
t2s = stoch.pivot_table(index=['variant', 'label'], columns='m', values='gap', aggfunc='median').round(4)
t2s.to_csv(f'{OUT}/rq2_gap_by_size.csv')
t2t = stoch.pivot_table(index=['variant', 'label'], columns='m', values='time', aggfunc='median').round(4)
t2t.to_csv(f'{OUT}/rq2_time_by_size.csv')

# Friedman + Wilcoxon-Holm on per-instance median gap (tuned methods; deterministic ones as is)
main = stoch[stoch.params != 'default']
S['friedman'] = {}; S['wilcoxon_vs_best'] = {}
for v in 'AB':
    M = main[main.variant == v].groupby(['instance', 'label']).gap.median().unstack().dropna(axis=1)
    if 'TCT' in M: M = M.drop(columns='TCT')
    st = friedmanchisquare(*[M[c] for c in M])
    mean_rank = M.rank(axis=1).mean().sort_values()
    S['friedman'][v] = dict(stat=round(float(st.statistic), 2), p=float(st.pvalue), k=M.shape[1], blocks=M.shape[0],
                            mean_rank=mean_rank.round(2).to_dict())
    bestm = mean_rank.index[0]; res = []
    for c in mean_rank.index[1:]:
        d = M[c] - M[bestm]
        p = 1.0 if np.allclose(d, 0) else wilcoxon(M[c], M[bestm], zero_method='zsplit').pvalue
        res.append([c, p, float((d > 1e-12).mean()), float((d < -1e-12).mean())])
    res.sort(key=lambda x: x[1]); k = len(res)
    for i, r in enumerate(res): r.append(min(1.0, max(r[1] * (k - i), res[i - 1][4] if i else 0)))   # Holm
    S['wilcoxon_vs_best'][v] = dict(reference=bestm, tests=[dict(method=r[0], p=r[1], p_holm=r[4],
                                    frac_worse=round(r[2], 3), frac_better=round(r[3], 3)) for r in res])

# Vargha-Delaney A12 per instance vs RA (variant A) and vs BLDM (variant B): P(method better)
def a12(x, y):
    x, y = np.asarray(x), np.asarray(y)
    return ((x[:, None] < y[None, :]).sum() + 0.5 * (x[:, None] == y[None, :]).sum()) / (len(x) * len(y))
A = []
for v, refm in (('A', 'RA'), ('B', 'ILS')):
    d = main[main.variant == v]
    for inst, di in d.groupby('instance'):
        ref = di[di.label == refm].gap
        for lab, dl in di.groupby('label'):
            if lab == refm or len(dl) < 2: continue
            A.append(dict(variant=v, reference=refm, label=lab, instance=inst, A12=a12(dl.gap, ref)))
A = pd.DataFrame(A)
t_a12 = A.groupby(['variant', 'reference', 'label']).A12.agg(['median', lambda s: (s > 0.56).mean() * 100,
                                                                lambda s: (s < 0.44).mean() * 100]).round(3)
t_a12.columns = ['median_A12', 'pct_instances_method_better(A12>0.56)', 'pct_instances_reference_better(A12<0.44)']
t_a12.to_csv(f'{OUT}/rq2_a12.csv')

# tuned vs default
dsub = ok[ok.instance.isin(ok[ok.params == 'default'].instance.unique()) & ok.method.isin(['RKGA', 'PSO', 'DE', 'GOA', 'HHO', 'RRG'])]
td = dsub.groupby(['variant', 'method', 'params']).agg(median_gap=('gap', 'median'), hit_pct=('hit', 'mean'),
                                                      below_tct_power_pct=('below_tct', 'mean')).round(4)
td['hit_pct'] *= 100; td['below_tct_power_pct'] *= 100
td.to_csv(f'{OUT}/tuned_vs_default.csv')

# anytime behaviour
cols = [c for c in ok.columns if c.startswith('var@')]
an = ok.dropna(subset=cols[:1])
rows = []
for c in cols:
    sdc = np.sqrt(an[c].clip(lower=0)); gp = (sdc - an.sd_best) / an.mean_row * 100
    for (v, lab), gg in gp.groupby([an.variant, an.label]):
        rows.append(dict(variant=v, label=lab, evals=int(c.split('@')[1]), median_gap=float(gg.median()),
                         hit_pct=float((gg <= 1e-7).mean() * 100)))
pd.DataFrame(rows).pivot_table(index=['variant', 'label'], columns='evals', values='median_gap').round(4).to_csv(f'{OUT}/anytime_gap.csv')
pd.DataFrame(rows).to_csv(f'{OUT}/anytime_long.csv', index=False)

# ---- RQ3: surrogate vs power in the full run ------------------------------------
rho = []
for inst, di in ok.groupby('instance'):
    if di['var'].nunique() > 3 and di.power.nunique() > 3:
        rho.append(spearmanr(di['var'], -di.power).correlation)
S['rq3_full_spearman_var_power'] = dict(median=round(float(np.median(rho)), 3), min=round(float(np.min(rho)), 3), n=len(rho))
bestvar = ok.loc[ok.groupby('instance')['var'].idxmin()].set_index('instance')
maxp = ok.groupby('instance').power.max()
S['rq3_full_power_regret_of_best_var_pct'] = dict(
    median=round(float(((1 - bestvar.power / maxp) * 100).median()), 4),
    max=round(float(((1 - bestvar.power / maxp) * 100).max()), 4),
    pct_zero=round(float(((1 - bestvar.power / maxp) * 100 < 1e-6).mean() * 100), 1))

# ---- RQ4: effect of instance features ---------------------------------------------
try:
    import statsmodels.formula.api as smf
    R4 = main[main.method.isin(['RA', 'BRA', 'ILS', 'RRG', 'RKGA', 'PSO', 'DE', 'GOA', 'HHO', 'RandomSearch'])]
    R4 = R4.groupby(['instance', 'variant', 'label', 'm', 'levels', 'kind', 'frac', 'set'], dropna=False).gap.median().reset_index()
    R4 = R4[R4.set.isin(['S1', 'S2'])]
    R4['y'] = np.log10(1e-4 + R4.gap); R4['logm'] = np.log2(R4.m)
    R4['frac'] = R4.frac.astype(float)
    mres = {}
    for v in 'AB':
        d = R4[R4.variant == v]
        md = smf.mixedlm('y ~ C(label, Treatment("RandomSearch")) * (logm + C(levels) + frac) + C(kind)', d,
                         groups=d['instance']).fit(reml=True)
        coef = md.summary().tables[1]
        coef.to_csv(f'{OUT}/rq4_mixedlm_{v}.csv')
        mres[v] = dict(n=int(len(d)), converged=bool(md.converged))
    S['rq4_mixedlm'] = mres
except Exception as e:
    S['rq4_mixedlm'] = f'failed: {e}'

# ---- power gains ---------------------------------------------------------------
pg = ok[ok.method.isin(['CP-SAT'])].groupby(['variant', 'levels']).power_gain.median().round(2)
S['cpsat_power_gain_median_pct'] = {f'{a}|{b}': float(v) for (a, b), v in pg.items()}
S['best_power_gain_over_tct_median_pct'] = round(float(((maxp / tct_p.reindex(maxp.index)) - 1).median() * 100), 2)

ok.to_csv(f'{OUT}/all_runs_with_gap.csv.gz', index=False, compression='gzip')
json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=1, default=float)
print(json.dumps(S, indent=1, default=float)[:4000])
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200)
print(t2)
print(t1); print(t1t)
