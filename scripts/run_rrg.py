"""Additional runs for the proposed method (variant B), same instances, seeds, budget and
measurements as scripts/run_full.py:
  RRG (tuned)            x 30 seeds, all test instances
  RRG (default)          x 30 seeds, default-parameter subset (as for the metaheuristics)
  BLDM+ILS (tuned kick)  x 30 seeds, the straightforward hybrid used as a baseline
Output: results/full_rrg/<instance>.csv (same columns as results/full).
Usage: python scripts/run_rrg.py [n_workers]
"""
import csv, json, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import run_full as rf
from pvrearrange import features, all_objectives, heuristics as H
from pvrearrange.power import TCTModel
from pvrearrange.regroup import rrg

OUT = 'results/full_rrg'
TUNED = json.load(open('results/tuned_params.json'))
DEFAULT_RRG = dict(rule='mixed', block=0.5)


def run_rrg(inst, seed, cfg):
    tr = []
    R = rrg(inst, rf.BUDGET, seed=seed, rule=cfg['rule'], block_iters=max(3, int(round(cfg['block'] * inst.n))), trace=tr)
    return R, tr


def solve_instance(spec):
    st, inst = rf.make(spec)
    path = f'{OUT}/{inst.name}.csv'
    if os.path.exists(path): return inst.name, 0.0
    t_inst = time.time(); model = TCTModel(); pcache = {}; ft = features(inst)
    base = dict(instance=inst.name, set=st, m=inst.m, n=inst.n, kind=inst.meta.get('kind', 'standard'),
                levels=inst.meta.get('levels', 'std'), frac=inst.meta.get('shaded_frac', ''),
                sigma=inst.meta.get('sigma', 0.0), n_values=ft['n_values'], cv=round(ft['cv'], 5),
                col_het=round(ft['col_heterogeneity'], 5), mean_row=inst.flat.sum() / inst.m)
    rows = []

    def record(method, params, seed, R, t, trace=None):
        assert inst.is_feasible(R, 'B')
        ob = all_objectives(inst, R)
        key = np.sort(np.sort(inst.flat[R], axis=1), axis=0).tobytes()
        if key not in pcache: pcache[key] = model.gmpp([inst.flat[r].tolist() for r in R])
        g = pcache[key]
        row = dict(base, variant='B', method=method, params=params, seed=seed, status='heuristic', time=round(t, 5),
                   power=round(g['P'], 3), peaks=g['peaks'], **{k: round(v, 5) for k, v in ob.items()})
        if trace:
            row['evals_to_best'] = rf.evals_to_best(trace)
            for c in rf.CHECK: row[f'var@{c}'] = round(rf.at(trace, c), 5)
        rows.append(row)

    timed = lambda f: (lambda t0: (f(), time.perf_counter() - t0))(time.perf_counter())
    default_subset = st == 'S1' and inst.m in (6, 12, 20) and inst.meta['levels'] in (3, 'continuous')
    b0 = H.bldm(inst)
    for seed in rf.SEEDS:
        (R, tr), t = timed(lambda: run_rrg(inst, seed, TUNED['RRG'])); record('RRG', 'tuned', seed, R, t, tr)
        if default_subset:
            (R, tr), t = timed(lambda: run_rrg(inst, seed, DEFAULT_RRG)); record('RRG', 'default', seed, R, t, tr)
        R, t = timed(lambda: H.ils(inst, 'B', 'var', rf.BUDGET, kick=TUNED['ILS']['kick'], seed=seed, R0=b0))
        record('BLDM+ILS', 'tuned', seed, R, t)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=rf.FIELDS); w.writeheader(); w.writerows(rows)
    os.replace(tmp, path)
    return inst.name, time.time() - t_inst


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    specs = rf.design()
    specs.sort(key=lambda sp: sp[1].get('m', 9) ** 2, reverse=True)
    nw = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    with Pool(nw) as pool:
        for k, (name, sec) in enumerate(pool.imap_unordered(solve_instance, specs), 1):
            print(f'{k}/{len(specs)} {name} {sec:.0f}s', flush=True)
