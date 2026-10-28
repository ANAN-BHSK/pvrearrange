"""Component analysis of RRG on the test instances (variant B, 10 seeds, 10,000 evaluations)."""
import json, sys, time
from multiprocessing import Pool
import numpy as np, pandas as pd
sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
import run_full as rf
from pvrearrange.regroup import rrg
T = json.load(open('results/tuned_params.json'))['RRG']
BI = lambda inst: max(3, int(round(T['block'] * inst.n)))
VARIANTS = {
    'RRG': lambda inst, s: rrg(inst, 10_000, s, rule=T['rule'], block_iters=BI(inst)),
    'no regrouping': lambda inst, s: rrg(inst, 10_000, s, rule=T['rule'], block_iters=BI(inst), regrouping=False),
    'no block steps': lambda inst, s: rrg(inst, 10_000, s, rule=T['rule'], block_iters=0),
    'sorted regrouping': lambda inst, s: rrg(inst, 10_000, s, rule='sorted', block_iters=BI(inst)),
    'mixed regrouping': lambda inst, s: rrg(inst, 10_000, s, rule='mixed', block_iters=BI(inst)),
    'random start': lambda inst, s: rrg(inst, 10_000, s, rule=T['rule'], block_iters=BI(inst), start='random'),
}
def job(spec):
    st, inst = rf.make(spec); out = []
    for name, fn in VARIANTS.items():
        for s in range(10):
            t = time.perf_counter(); R = fn(inst, s)
            out.append(dict(instance=inst.name, set=st, m=inst.m, variant_name=name, seed=s,
                            var=float(inst.row_sums(R).var()), time=time.perf_counter() - t))
    return out
if __name__ == '__main__':
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        res = p.map(job, rf.design(), chunksize=1)
    pd.DataFrame([r for x in res for r in x]).to_csv('results/ablation_rrg.csv', index=False)
