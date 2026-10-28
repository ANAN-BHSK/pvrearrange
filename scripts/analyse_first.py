"""Summaries of the first experiment.

gap = (sd(S) - sd*(S)) / mean row sum x 100, where sd* is the best standard deviation
found by any method (CP-SAT included) for the same instance and variant. A gap of 0
means the method found the best known solution (the proven optimum where CP-SAT
reports OPTIMAL)."""
import sys
import numpy as np
import pandas as pd

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'results/first_results.csv')
df['levels'] = df['levels'].astype(str)
df = df.dropna(subset=['var'])
df['sd'] = np.sqrt(df['var'])
best = df.groupby(['instance', 'variant'])['sd'].transform('min')
# every variant-A solution is feasible for variant B, so B's best known includes A's
bestA = df[df.variant == 'A'].groupby('instance')['sd'].min()
isB = df.variant == 'B'
best[isB] = np.minimum(best[isB], df.loc[isB, 'instance'].map(bestA))
df['gap'] = (df['sd'] - best) / df['mean_row'] * 100
df['hit'] = df['gap'] < 1e-6

pd.set_option('display.width', 200)
print('Instances:', df['instance'].nunique())

print('\n== CP-SAT: proven optimal (%) and median time (s) by size x levels ==')
cp = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'results/first_results.csv')
cp = cp[cp.method == 'CP-SAT']; cp['levels'] = cp['levels'].astype(str)
cp['opt'] = cp.status == 'OPTIMAL'
print(cp.pivot_table(index=['variant', 'm'], columns='levels', values='opt', aggfunc='mean').mul(100).round(0))
print(cp.pivot_table(index=['variant', 'm'], columns='levels', values='time', aggfunc='median').round(2))

for v in 'AB':
    d = df[df.variant == v]
    print(f'\n== Variant {v}: median gap (% of mean row sum) by method x levels ==')
    print(d.pivot_table(index='method', columns='levels', values='gap', aggfunc='median').round(3))
    print(f'\n== Variant {v}: share of instances where method found the best known (%) ==')
    print(d.pivot_table(index='method', columns='levels', values='hit', aggfunc='mean').mul(100).round(0))
    print(f'\n== Variant {v}: median time (s) by method x size ==')
    print(d.pivot_table(index='method', columns='m', values='time', aggfunc='median').round(4))

# value of free assignment: best B vs best A
bA = df[df.variant == 'A'].groupby('instance')['sd'].min()
bB_own = df[df.variant == 'B'].groupby('instance')['sd'].min()
bB = np.minimum(bB_own, bA)
print('\nInstances where B-specific methods did worse than the best A solution:',
      int((bB_own > bA + 1e-9).sum()), 'of', len(bA))
mr = df.groupby('instance')['mean_row'].first()
print('\n== Best sd(S) as % of mean row sum: A vs B (median over instances) ==')
print(pd.DataFrame({'A': (bA / mr * 100), 'B': (bB / mr * 100),
                    'levels': df.groupby('instance')['levels'].first()}).groupby('levels').median().round(3))
df.to_csv('results/first_results_with_gap.csv', index=False)
