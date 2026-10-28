"""First experiment: exact (CP-SAT) vs classical heuristics on random instances.

Factors: size {6,9,12,16,20}^2, shadow kind {rect, cluster, scatter},
shaded levels {1, 3, continuous}, 2 seeds, shaded fraction 0.3.
Primary objective: variance of row sums. Stochastic heuristics use one seed here;
the full protocol uses 30.

Usage: python scripts/run_first.py out.csv [time_limit]
"""
import csv, sys, time
import numpy as np
sys.path.insert(0, '.')
from pvrearrange import random_instance, features, all_objectives, exact, heuristics as H

out = sys.argv[1] if len(sys.argv) > 1 else 'first_results.csv'
TL = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0

sizes = [6, 9, 12, 16, 20]
kinds = ['rect', 'cluster', 'scatter']
levels = [1, 3, 'continuous']
seeds = [0, 1]

A_methods = {
    'TCT':      lambda I: I.identity(),
    'Greedy':   lambda I: H.greedy_columns(I),
    'RA':       lambda I: H.ra(I, restarts=30, seed=0),
    'BRA':      lambda I: H.block_ra(I, rule='random', then_ra=False, iters=200, seed=0),
    'BRAVE+RA': lambda I: H.block_ra(I, rule='brave', seed=0),
    'ILS':      lambda I: H.ils(I, 'A', budget_evals=10_000, seed=0),
}
B_methods = {
    'BLDM':     lambda I: H.bldm(I),
    'BWSA':     lambda I: H.bwsa(I),
    'SubsetSum+Munkres': lambda I: H.subset_sum_rows(I, resolution=50.0) if I.m <= 9 else None,
    'ILS':      lambda I: H.ils(I, 'B', budget_evals=10_000, seed=0),
}

fields = ['instance', 'm', 'n', 'kind', 'levels', 'seed', 'n_values', 'cv', 'col_heterogeneity',
          'variant', 'method', 'status', 'time', 'var', 'L1', 'range', 'bot', 'bound_var', 'mean_row']
with open(out, 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
    for s in sizes:
        for kind in kinds:
            for L in levels:
                for seed in seeds:
                    inst = random_instance(s, s, kind, 0.3, levels=L, seed=seed)
                    ft = features(inst)
                    base = dict(instance=inst.name, m=s, n=s, kind=kind, levels=L, seed=seed,
                                n_values=ft['n_values'], cv=round(ft['cv'], 5),
                                col_heterogeneity=round(ft['col_heterogeneity'], 5),
                                mean_row=inst.flat.sum() / s)
                    for variant, methods in (('A', A_methods), ('B', B_methods)):
                        best_heur = None
                        for name, fn in methods.items():
                            t0 = time.perf_counter(); R = fn(inst); t = time.perf_counter() - t0
                            if R is None: continue
                            assert inst.is_feasible(R, variant if name != 'TCT' else 'A')
                            ob = all_objectives(inst, R)
                            if best_heur is None or ob['var'] < all_objectives(inst, best_heur)['var']:
                                best_heur = R
                            w.writerow({**base, 'variant': variant, 'method': name, 'status': 'heuristic',
                                        'time': round(t, 5), **{k: round(v, 4) for k, v in ob.items()}})
                        r = exact.solve_cpsat(inst, variant, 'var', time_limit=TL, workers=2, hint=best_heur)
                        if r['R'] is not None:
                            ob = all_objectives(inst, r['R'])
                            w.writerow({**base, 'variant': variant, 'method': 'CP-SAT', 'status': r['status'],
                                        'time': round(r['time'], 3), **{k: round(v, 4) for k, v in ob.items()},
                                        'bound_var': round(r['bound'], 4)})
                        else:
                            w.writerow({**base, 'variant': variant, 'method': 'CP-SAT', 'status': r['status'],
                                        'time': round(r['time'], 3)})
                        fh.flush()
                    print(inst.name, flush=True)
