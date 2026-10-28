import json
import numpy as np
import pandas as pd

runs = pd.read_csv('results/replication_runs.csv')
ref = pd.read_csv('results/replication_reference.csv')
pub = json.load(open('results/premkumar_tables.json'))
tab_of = dict(zip(['none', 'uneven_row', 'uneven_column', 'diagonal', 'short_narrow', 'short_wide',
                   'long_narrow', 'long_wide', 'central'], map(str, range(5, 14))))
pd.set_option('display.width', 220)

opt = ref[ref.method.str.startswith('CP-SAT[L1]')].groupby('case')['L1'].min()
tct = ref[ref.method == 'TCT'].set_index('case')['power_W']
runs['opt'] = runs.case.map(opt)
runs['hit'] = runs.L1 <= runs.opt + 1e-6
runs['below_tct'] = runs.power_W < runs.case.map(tct) - 0.5

print('== Share of runs reaching the L1 optimum (%), 20 runs x 9 cases ==')
print(runs.pivot_table(index='method', columns='variant', values='hit', aggfunc='mean').mul(100).round(1))
print('\n== Share of runs with power below unreconfigured TCT (%) ==')
print(runs.pivot_table(index='method', columns='variant', values='below_tct', aggfunc='mean').mul(100).round(1))

print('\n== Per case, variant B: optimum hit rate (%) ==')
print(runs[runs.variant == 'B'].pivot_table(index='case', columns='method', values='hit', aggfunc='mean').mul(100).round(0))

print('\n== Power (W): published best-of-20 vs our best-of-20 and median, variant B ==')
rows = []
for c in opt.index:
    t = pub[tab_of[c]]
    for m in ['PSO', 'DE', 'HHO']:
        d = runs[(runs.case == c) & (runs.method == m) & (runs.variant == 'B')]
        rows.append(dict(case=c, method=m, published=t[m][0], ours_best=d.power_W.max(),
                         ours_median=d.power_W.median(), ours_worst=d.power_W.min()))
cmp_ = pd.DataFrame(rows)
print(cmp_.round(1).to_string(index=False))

print('\n== Reference methods: L1 and power ==')
r2 = ref.copy(); r2['opt'] = r2.case.map(opt); r2['hit'] = r2.L1 <= r2.opt + 1e-6
print(r2.pivot_table(index='method', values=['hit'], aggfunc='mean').mul(100).round(0))
print(r2.groupby('method')['time_s'].max().round(4))
print('\n== Median evaluations to reach final best, by method (variant B) ==')
print(runs[runs.variant == 'B'].groupby('method')['evals_to_best'].median())
cmp_.to_csv('results/replication_vs_published.csv', index=False)
