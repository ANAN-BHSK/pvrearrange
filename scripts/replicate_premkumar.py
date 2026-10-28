"""Replication of Premkumar et al. (2026): nine 6x6 shading cases, objective L1,
population 40 x 250 iterations = 10,000 evaluations, 20 independent runs.
Both encodings are run: variant B (permutation of all 36 modules, as in the paper's
formulation) and variant A (within-column, as in the paper's switching hardware).

Output: results/replication_runs.csv (one row per run) and
        results/replication_reference.csv (exact optimum and deterministic methods).
"""
import csv, sys, time, json
import numpy as np
sys.path.insert(0, '.')
from pvrearrange import premkumar_instances, exact, heuristics as H, all_objectives
from pvrearrange.metaheuristics import METAHEURISTICS
from pvrearrange.power import TCTModel

RUNS, BUDGET, POP = 20, 10_000, 40
model = TCTModel()
insts = premkumar_instances()

with open('results/replication_reference.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'method', 'variant', 'L1', 'range', 'power_W', 'time_s', 'status'])
    for inst in insts:
        c = inst.meta['pattern']
        def put(name, v, R, t, st=''):
            ob = all_objectives(inst, R)
            w.writerow([c, name, v, round(ob['L1'], 3), round(ob['range'], 3),
                        round(model.power(inst, R), 2), round(t, 5), st])
        put('TCT', 'A', inst.identity(), 0.0)
        for v in 'AB':
            for ob in ('L1', 'range'):
                r = exact.solve_cpsat(inst, v, ob, time_limit=60, workers=2, resolution=100)
                put(f'CP-SAT[{ob}]', v, r['R'], r['time'], r['status'])
        t0 = time.perf_counter(); R = H.ra(inst, restarts=30, seed=0); put('RA', 'A', R, time.perf_counter() - t0)
        t0 = time.perf_counter(); R = H.block_ra(inst, rule='random', then_ra=False, seed=0); put('BRA', 'A', R, time.perf_counter() - t0)
        t0 = time.perf_counter(); R = H.bldm(inst); put('BLDM', 'B', R, time.perf_counter() - t0)
        t0 = time.perf_counter(); R = H.bwsa(inst); put('BWSA', 'B', R, time.perf_counter() - t0)
        print('reference', c, flush=True)

with open('results/replication_runs.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['case', 'method', 'variant', 'seed', 'L1', 'range', 'power_W', 'evals_to_best', 'time_s'])
    for inst in insts:
        c = inst.meta['pattern']
        for name, fn in METAHEURISTICS.items():
            for v in 'AB':
                for seed in range(RUNS):
                    t0 = time.perf_counter()
                    r = fn(inst, v, 'L1', BUDGET, pop=POP, seed=seed)
                    t = time.perf_counter() - t0
                    # evaluations needed to reach the final best value
                    e_best = next(e for e, b in r['trace'] if b <= r['value'] + 1e-9)
                    ob = all_objectives(inst, r['R'])
                    w.writerow([c, name, v, seed, round(ob['L1'], 3), round(ob['range'], 3),
                                round(model.power(inst, r['R']), 2), e_best, round(t, 4)])
            print('runs', c, name, flush=True)
