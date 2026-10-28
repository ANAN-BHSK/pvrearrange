"""Full test run (protocol Sec. 7 and 10, reduced design; see README).

Test instances (seeds disjoint from tuning):
  S1 main factorial : sizes {6,9,12,16,20,30} x kinds {rect,cluster,scatter} x levels {1,3,cont} x frac 0.25
  S2 shaded fraction: sizes {9,16} x kinds x levels {3,cont} x frac {0.10,0.40,0.60}
  S3 module ageing  : sizes {9,16} x kinds x levels 3 x frac 0.25 x sigma {0.02,0.05}
  S4 standard shapes: the eight shaded patterns of Premkumar et al. (2026) scaled to 9x9 and 20x20
Methods (both variants where applicable), objective = variance of row sums, budget 10,000 evals:
  deterministic once: TCT, Greedy(A), BLDM(B), BWSA(B), CP-SAT (1 thread, 60 s, no warm start)
  stochastic x 30 seeds: RA(A), BRA(A), BRAVE+RA(A), ILS, RandomSearch, RKGA, PSO, DE, GOA, HHO (tuned)
  default-parameter metaheuristics x 30 seeds on a subset (S1, sizes 6/12/20, levels 3/cont)
Usage: python scripts/run_full.py [n_workers]
"""
import csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '.')
from pvrearrange import random_instance, scaled_standard, features, all_objectives, exact, heuristics as H
from pvrearrange.instance import PREMKUMAR_6x6
from pvrearrange.metaheuristics import random_search, rkga, pso, de, goa, hho
from pvrearrange.power import TCTModel

SEEDS = range(30)
BUDGET = 10_000
CHECK = (1000, 2500, 5000, 10000)
OUT = 'results/full'
TUNED = json.load(open('results/tuned_params.json'))
DEFAULTS = dict(RKGA=dict(pop=40, elite=0.2, mutants=0.1, bias=0.7), PSO=dict(pop=40, w0=0.9, w1=0.4, c1=2.0, c2=2.0, vmax=0.2),
                DE=dict(pop=40, F=0.5, CR=0.9), GOA=dict(pop=40, cmax=1.0, cmin=1e-5, fs=0.5, ls=1.5), HHO=dict(pop=40))
FIELDS = ['instance', 'set', 'm', 'n', 'kind', 'levels', 'frac', 'sigma', 'n_values', 'cv', 'col_het',
          'variant', 'method', 'params', 'seed', 'status', 'time', 'evals_to_best',
          'var', 'L1', 'range', 'bot', 'bound_var', 'power', 'peaks', 'mean_row'] + [f'var@{c}' for c in CHECK]


def design():
    D = []
    for s in (6, 9, 12, 16, 20, 30):
        for k in ('rect', 'cluster', 'scatter'):
            for L in (1, 3, 'continuous'):
                D.append(('S1', dict(m=s, n=s, kind=k, shaded_frac=0.25, levels=L, seed=3000 + s)))
    for s in (9, 16):
        for k in ('rect', 'cluster', 'scatter'):
            for L in (3, 'continuous'):
                for fr in (0.10, 0.40, 0.60):
                    D.append(('S2', dict(m=s, n=s, kind=k, shaded_frac=fr, levels=L, seed=3100 + s)))
    for s in (9, 16):
        for k in ('rect', 'cluster', 'scatter'):
            for sg in (0.02, 0.05):
                D.append(('S3', dict(m=s, n=s, kind=k, shaded_frac=0.25, levels=3, sigma=sg, seed=3200 + s)))
    for s in (9, 20):
        for p in PREMKUMAR_6x6:
            if p != 'none': D.append(('S4', dict(pattern=p, m=s, n=s)))
    return D


def make(spec):
    st, a = spec
    if st == 'S4': return st, scaled_standard(a['pattern'], a['m'], a['n'])
    a = dict(a); m, n = a.pop('m'), a.pop('n')
    return st, random_instance(m, n, **a)


def mh_call(name, inst, v, seed, cfg):
    c = dict(cfg)
    if name == 'PSO': c['w'] = (c.pop('w0'), c.pop('w1'))
    f = dict(RandomSearch=random_search, RKGA=rkga, PSO=pso, DE=de, GOA=goa, HHO=hho)[name]
    r = f(inst, v, 'var', BUDGET, seed=seed, **c)
    return r['R'], r['trace']


def ra_traced(inst, seed, rule=None):
    """RA with random restarts until the evaluation budget is used; returns (R, trace)."""
    rng = np.random.default_rng(seed); B = H.Budget(BUDGET); best, bv, trace = None, np.inf, []
    first = True
    while B.left:
        R0 = inst.identity() if first else H.random_solution(inst, 'A', rng); first = False
        R = H.ra_single(inst, R0, budget=B)
        v = inst.row_sums(R).var()
        if v < bv: best, bv = R, v
        trace.append((B.used, bv))
    return best, trace


def at(trace, e):
    vals = [b for u, b in trace if u <= e]
    return vals[-1] if vals else np.nan


def evals_to_best(trace):
    fin = trace[-1][1]
    return next(u for u, b in trace if b <= fin + 1e-9)


def solve_instance(spec):
    st, inst = make(spec)
    path = f'{OUT}/{inst.name}.csv'
    if os.path.exists(path): return inst.name, 0.0
    t_inst = time.time()
    model = TCTModel(); pcache = {}
    ft = features(inst)
    base = dict(instance=inst.name, set=st, m=inst.m, n=inst.n, kind=inst.meta.get('kind', 'standard'),
                levels=inst.meta.get('levels', 'std'), frac=inst.meta.get('shaded_frac', ''),
                sigma=inst.meta.get('sigma', 0.0), n_values=ft['n_values'], cv=round(ft['cv'], 5),
                col_het=round(ft['col_heterogeneity'], 5), mean_row=inst.flat.sum() / inst.m)
    rows = []

    def record(variant, method, params, seed, R, t, status='heuristic', trace=None, bound=None):
        ob = all_objectives(inst, R)
        key = np.sort(np.sort(inst.flat[R], axis=1), axis=0).tobytes()   # power depends on row compositions
        if key not in pcache: pcache[key] = model.gmpp([inst.flat[r].tolist() for r in R])
        g = pcache[key]
        row = dict(base, variant=variant, method=method, params=params, seed=seed, status=status,
                   time=round(t, 5), power=round(g['P'], 3), peaks=g['peaks'],
                   **{k: round(v, 5) for k, v in ob.items()})
        if bound is not None: row['bound_var'] = round(bound, 5)
        if trace:
            row['evals_to_best'] = evals_to_best(trace)
            for c in CHECK: row[f'var@{c}'] = round(at(trace, c), 5)
        rows.append(row)

    timed = lambda f: (lambda t0: (f(), time.perf_counter() - t0))(time.perf_counter())
    R, t = timed(lambda: inst.identity()); record('A', 'TCT', '-', 0, R, t)
    R, t = timed(lambda: H.greedy_columns(inst)); record('A', 'Greedy', '-', 0, R, t)
    R, t = timed(lambda: H.bldm(inst)); record('B', 'BLDM', '-', 0, R, t)
    R, t = timed(lambda: H.bwsa(inst)); record('B', 'BWSA', '-', 0, R, t)
    for v in 'AB':
        r = exact.solve_cpsat(inst, v, 'var', time_limit=60, workers=1)
        if r['R'] is not None:
            record(v, 'CP-SAT', '-', 0, r['R'], r['time'], r['status'], bound=r['bound'])
        else:
            rows.append(dict(base, variant=v, method='CP-SAT', params='-', seed=0, status=r['status'], time=round(r['time'], 3)))

    default_subset = st == 'S1' and inst.m in (6, 12, 20) and inst.meta['levels'] in (3, 'continuous')
    for seed in SEEDS:
        (R, tr), t = timed(lambda: ra_traced(inst, seed)); record('A', 'RA', '-', seed, R, t, trace=tr)
        R, t = timed(lambda: H.block_ra(inst, rule='random', then_ra=False, iters=BUDGET, seed=seed)); record('A', 'BRA', '-', seed, R, t)
        R, t = timed(lambda: H.block_ra(inst, rule='brave', seed=seed)); record('A', 'BRAVE+RA', '-', seed, R, t)
        for v in 'AB':
            R, t = timed(lambda: H.ils(inst, v, 'var', BUDGET, kick=TUNED['ILS']['kick'], seed=seed)); record(v, 'ILS', 'tuned', seed, R, t)
            (R, tr), t = timed(lambda: mh_call('RandomSearch', inst, v, seed, {})); record(v, 'RandomSearch', '-', seed, R, t, trace=tr)
            for name in ('RKGA', 'PSO', 'DE', 'GOA', 'HHO'):
                (R, tr), t = timed(lambda: mh_call(name, inst, v, seed, TUNED[name])); record(v, name, 'tuned', seed, R, t, trace=tr)
                if default_subset:
                    (R, tr), t = timed(lambda: mh_call(name, inst, v, seed, DEFAULTS[name])); record(v, name, 'default', seed, R, t, trace=tr)

    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)
    return inst.name, time.time() - t_inst


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    specs = design()
    cost = lambda sp: sp[1].get('m', 9) ** 2 * (2 if sp[1].get('levels') == 'continuous' else 1)
    specs.sort(key=cost, reverse=True)                     # largest first for load balance
    print(len(specs), 'instances', flush=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(nw) as pool:
        for k, (name, sec) in enumerate(pool.imap_unordered(solve_instance, specs), 1):
            print(f'{k}/{len(specs)} {name} {sec:.0f}s', flush=True)
