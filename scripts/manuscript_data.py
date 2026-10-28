"""Numbers for the manuscript: report_data.json + replication study."""
import json, numpy as np, pandas as pd
D = json.load(open('results/analysis/report_data.json'))
runs = pd.read_csv('results/replication_runs.csv'); ref = pd.read_csv('results/replication_reference.csv')
opt = ref[ref.method == 'CP-SAT[L1]'].groupby('case').L1.min(); tct = ref[ref.method == 'TCT'].set_index('case').power_W
runs['hit'] = runs.L1 <= runs.case.map(opt) + 1e-6; runs['below'] = runs.power_W < runs.case.map(tct) - 0.5
rep = {}
for (m, v), g in runs.groupby(['method', 'variant']):
    rep.setdefault(m, {})[v] = dict(hit=float(g.hit.mean() * 100), below=float(g.below.mean() * 100), evals_med=float(g.evals_to_best.median()))
r = ref.copy(); r['hit'] = r.L1 <= r.case.map(opt) + 1e-6
refm = {m: dict(hit=float(g.hit.mean() * 100), tmax=float(g.time_s.max())) for m, g in r.groupby('method')}
pub = json.load(open('results/premkumar_tables.json'))
n = sum(1 for t in pub.values() for a in t if a != 'TCT')
nb = sum(1 for t in pub.values() for a, v in t.items() if a != 'TCT' and v[0] < t['TCT'][0] - 0.01)
c = pd.read_csv('results/replication_vs_published.csv'); c = c[c.case != 'none']
D['replication'] = dict(runs=rep, reference=refm, published_below_tct=nb, published_total=n,
                        worst_above_pub=int((c.ours_worst > c.published * 1.01).sum()), compared=int(len(c)),
                        tct_model_err_max=float((ref[ref.method == 'TCT'].set_index('case').power_W /
                                                 pd.Series({k: v['TCT'][0] for k, v in pub.items()}).rename(
                                                     {'5': 'none', '6': 'uneven_row', '7': 'uneven_column', '8': 'diagonal',
                                                      '9': 'short_narrow', '10': 'short_wide', '11': 'long_narrow',
                                                      '12': 'long_wide', '13': 'central'}) - 1).abs().max() * 100))
json.dump(D, open('results/analysis/manuscript_data.json', 'w'), indent=1)
print(json.dumps(D['replication'], indent=1)[:1500])

# repeat of sensitivity/RQ3 and tuning on a second computer (run_all.py --mode full)
import os
if os.path.exists('results/second_machine/report_data.json'):
    R2 = json.load(open('results/second_machine/report_data.json'))
    mi = json.load(open('results/second_machine/machine_info.json'))
    D['second_machine'] = dict(rq3_spearman=R2['rq3_spearman'], sensitivity=R2['sensitivity'], rq3_regret=R2['rq3_regret'],
                               tuned_identical=json.load(open('results/second_machine/tuned_params.json')) == D['tuned'],
                               machine=dict(os=mi['os'], cpu_count=mi['cpu_count'], ram_gb=mi.get('ram_gb'), python=mi['python']))
    json.dump(D, open('results/analysis/manuscript_data.json', 'w'), indent=1)
    print('second machine:', D['second_machine']['tuned_identical'])

# effect of the long CP-SAT runs and best-of-runs optimality on proven instances
ok = pd.read_csv('results/analysis/all_runs_with_gap.csv.gz')
prev = 'results/analysis_before_long_cpsat/all_runs_with_gap.csv.gz'
if os.path.exists(prev):
    o = pd.read_csv(prev)
    b1 = ok.groupby(['instance', 'variant']).sd_best.first(); b0 = o.groupby(['instance', 'variant']).sd_best.first()
    mr = ok.groupby('instance').mean_row.first()
    dd = (b0 - b1) / b1.index.get_level_values(0).map(mr).values * 100
    D['long_improved'] = dict(pairs=int((dd > 1e-9).sum()), max=float(dd.max()), median=float(dd[dd > 1e-9].median()))
pr = ok[ok.proven]
D['best_of_runs'] = {f'{v}_{m}': float((pr[(pr.variant == v) & (pr.label == m)].groupby('instance').gap.min() < 1e-7).mean() * 100)
                     for v in 'AB' for m in ['BRA', 'RA', 'ILS', 'RKGA', 'BLDM', 'RRG', 'BLDM+ILS'] if len(pr[(pr.variant == v) & (pr.label == m)])}
a = ok[ok.variant == 'A'].groupby('instance')['var'].min()
lgp = 'results/cpsat_long_all.csv'
if os.path.exists(lgp):
    lg = pd.read_csv(lgp); la = lg[lg.variant == 'A'].set_index('instance').best_var
    a = np.minimum(a, la.reindex(a.index).fillna(np.inf))
bh = ok[(ok.variant == 'B') & (ok.method != 'CP-SAT')].groupby('instance')['var'].min()
D['a_beats_b'] = int((a < bh.reindex(a.index) - 1e-6 * (1 + a)).sum())
json.dump(D, open('results/analysis/manuscript_data.json', 'w'), indent=1)
print(D['long_improved'], D['best_of_runs'], D['a_beats_b'])

# low Spearman correlations between variance and power on the full set
from scipy.stats import spearmanr
low = []
for inst, di in ok.groupby('instance'):
    if di['var'].nunique() > 3 and di.power.nunique() > 3:
        r = spearmanr(di['var'], -di.power).correlation
        if r < 0.6: low.append(dict(instance=inst, rho=float(r), m=int(di.m.iloc[0]), n_power=int(di.power.nunique())))
D['rq3_low_rho'] = dict(n=len(low), negative=sum(x['rho'] < 0 for x in low), max_m=max(x['m'] for x in low) if low else None,
                        max_distinct_power=max(x['n_power'] for x in low) if low else None)
abl = 'results/ablation_rrg.csv'
if os.path.exists(abl):
    a = pd.read_csv(abl)
    sb = ok[ok.variant == 'B'].groupby('instance').sd_best.first(); mr = ok.groupby('instance').mean_row.first()
    a['gap'] = ((np.sqrt(a['var'].clip(lower=0)) - a.instance.map(sb)) / a.instance.map(mr) * 100).clip(lower=0)
    a['hit'] = a.gap <= 1e-7
    g = a.groupby('variant_name')
    D['ablation'] = [dict(name=k, median_gap=float(x.gap.median()), q90_gap=float(x.gap.quantile(0.9)), mean_gap=float(x.gap.mean()),
                          hit_pct=float(x.hit.mean() * 100), median_time=float(x.time.median())) for k, x in g]
json.dump(D, open('results/analysis/manuscript_data.json', 'w'), indent=1)
print(D['rq3_low_rho'])
