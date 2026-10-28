"""Parameter tuning on training instances with iterated racing (pvrearrange.racing).

Training instances (disjoint seeds from the test set): sizes 6x6 and 9x9, three shadow
kinds, 3 levels and continuous, shaded fraction 0.25 and 0.40; both variants; two run
seeds per (instance, variant). Objective: variance of row sums, budget 10,000 evaluations.
Output: results/tuned_params.json and results/tuning_history.json
"""
import json, sys, time
import numpy as np
sys.path.insert(0, '.')
from pvrearrange import random_instance, heuristics as H, OBJECTIVES
from pvrearrange.metaheuristics import rkga, pso, de, goa, hho
from pvrearrange.racing import Param, irace
from pvrearrange.regroup import rrg

BUDGET_EVALS = 10_000
TUNE_RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 400
which = sys.argv[2].split(',') if len(sys.argv) > 2 else ['RKGA', 'PSO', 'DE', 'GOA', 'HHO', 'ILS']

train = []
for s in (6, 9):
    for kind in ('rect', 'cluster', 'scatter'):
        for L in (3, 'continuous'):
            for fr in (0.25, 0.40):
                train.append(random_instance(s, s, kind, fr, levels=L, seed=2000 + s))
stream = [(inst, v, rs) for inst in train for v in 'AB' for rs in (11, 12)]

SPACES = {
    'RKGA': ({'pop': Param('int', 10, 200, log=True), 'elite': Param('real', 0.05, 0.35),
              'mutants': Param('real', 0.0, 0.3), 'bias': Param('real', 0.5, 0.95)},
             dict(pop=40, elite=0.2, mutants=0.1, bias=0.7)),
    'PSO': ({'pop': Param('int', 10, 200, log=True), 'w0': Param('real', 0.4, 1.0), 'w1': Param('real', 0.0, 0.6),
             'c1': Param('real', 0.0, 3.0), 'c2': Param('real', 0.0, 3.0), 'vmax': Param('real', 0.01, 1.0, log=True)},
            dict(pop=40, w0=0.9, w1=0.4, c1=2.0, c2=2.0, vmax=0.2)),
    'DE': ({'pop': Param('int', 10, 200, log=True), 'F': Param('real', 0.1, 1.2), 'CR': Param('real', 0.0, 1.0)},
           dict(pop=40, F=0.5, CR=0.9)),
    'GOA': ({'pop': Param('int', 10, 100, log=True), 'cmax': Param('real', 0.01, 2.0, log=True),
             'cmin': Param('real', 1e-6, 1e-1, log=True), 'fs': Param('real', 0.1, 1.0), 'ls': Param('real', 0.5, 3.0)},
            dict(pop=40, cmax=1.0, cmin=1e-5, fs=0.5, ls=1.5)),
    'HHO': ({'pop': Param('int', 5, 100, log=True)}, dict(pop=40)),
    'ILS': ({'kick': Param('int', 1, 15)}, dict(kick=3)),
    # proposed method, variant B only: regrouping rule and block-step patience (x n columns)
    'RRG': ({'rule': Param('cat', values=['sorted', 'random', 'mixed']), 'block': Param('real', 0.1, 2.0, log=True)},
            dict(rule='mixed', block=0.5)),
}


def call(alg, cfg, inst, v, rs):
    if alg == 'RKGA': return rkga(inst, v, 'var', BUDGET_EVALS, seed=rs, **cfg)['value']
    if alg == 'PSO':
        c = dict(cfg); w = (c.pop('w0'), c.pop('w1'))
        return pso(inst, v, 'var', BUDGET_EVALS, seed=rs, w=w, **c)['value']
    if alg == 'DE': return de(inst, v, 'var', BUDGET_EVALS, seed=rs, **cfg)['value']
    if alg == 'GOA': return goa(inst, v, 'var', BUDGET_EVALS, seed=rs, **cfg)['value']
    if alg == 'HHO': return hho(inst, v, 'var', BUDGET_EVALS, seed=rs, **cfg)['value']
    if alg == 'RRG':
        R = rrg(inst, BUDGET_EVALS, seed=rs, rule=cfg['rule'], block_iters=max(3, int(round(cfg['block'] * inst.n))))
        return OBJECTIVES['var'](inst.row_sums(R))
    if alg == 'ILS':
        R = H.ils(inst, v, 'var', BUDGET_EVALS, kick=cfg['kick'], seed=rs)
        return OBJECTIVES['var'](inst.row_sums(R))


try:
    tuned = json.load(open('results/tuned_params.json')); hist_all = json.load(open('results/tuning_history.json'))
except FileNotFoundError:
    tuned, hist_all = {}, {}
for alg in which:
    space, default = SPACES[alg]
    t0 = time.time(); print(alg, flush=True)
    st = [x for x in stream if x[1] == 'B'] if alg == 'RRG' else stream
    best, hist = irace(space, lambda c, x: call(alg, c, *x), st, TUNE_RUNS, seed=1, default=default,
                       log=lambda s: print(s, flush=True))
    tuned[alg] = best; hist_all[alg] = dict(history=hist, default=default, seconds=round(time.time() - t0, 1))
    json.dump(tuned, open('results/tuned_params.json', 'w'), indent=1)
    json.dump(hist_all, open('results/tuning_history.json', 'w'), indent=1, default=str)
    print(alg, 'done in', round(time.time() - t0), 's ->', best, flush=True)
