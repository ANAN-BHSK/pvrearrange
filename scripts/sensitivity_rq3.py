"""Sensitivity of conclusions to the electrical model (protocol Sec. 8) and
agreement between surrogate objectives and array power (RQ3).

Instances: the nine 6x6 cases of Premkumar et al. (2026), the same patterns scaled
to 9x9, and random 6x6 / 9x9 arrays with 2-4 shaded levels (plus 6x6 continuous), for
which CP-SAT proves optimality quickly. Variant A throughout.

For each instance:
  * CP-SAT optimum under each surrogate (var, L1, range, bot);
  * power of TCT and of each surrogate optimum under five electrical models;
  * best power found by swap local search on power (H7) started from TCT and from the
    best surrogate optimum (base model only) -> P_best for regret;
  * Spearman correlation between each surrogate and power over a pool of 300 solutions
    (100 random, 200 perturbations of the var-optimum), base model only.
"""
import csv, sys, time
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '.')
from pvrearrange import premkumar_instances, scaled_standard, random_instance, OBJECTIVES, exact, heuristics as H
from pvrearrange.instance import PREMKUMAR_6x6
from pvrearrange.power import TCTModel

MODELS = {
    'base':    {},
    'cells56': dict(cells=56, beta_rel=-0.004),
    'vbyp0.3': dict(vbyp=-0.3),
    'vbyp0.7': dict(vbyp=-0.7),
    'noct45':  dict(noct=45.0),
}
models = {k: TCTModel(v) for k, v in MODELS.items()}
base = models['base']


def power_ls(inst, model, R0, max_evals=2000, seed=0):
    """Swap local search on power, variant A, first improvement."""
    rng = np.random.default_rng(seed); m, n = inst.m, inst.n
    R = R0.copy(); cur = model.power(inst, R); ev = 1; improved = True
    while improved and ev < max_evals:
        improved = False
        for j in rng.permutation(n):
            for a in range(m):
                for b in range(a + 1, m):
                    if ev >= max_evals or inst.flat[R[a, j]] == inst.flat[R[b, j]]: continue
                    R[a, j], R[b, j] = R[b, j], R[a, j]
                    p = model.power(inst, R); ev += 1
                    if p > cur + 1e-6: cur = p; improved = True
                    else: R[a, j], R[b, j] = R[b, j], R[a, j]
    return R, cur


def pool(inst, Ropt, rng, n_rand=100, n_pert=200):
    sols = [H.random_solution(inst, 'A', rng) for _ in range(n_rand)]
    m, n = inst.m, inst.n
    for t in range(n_pert):
        R = Ropt.copy()
        for _ in range(1 + t % 20):
            j = rng.integers(n); a, b = rng.choice(m, 2, replace=False)
            R[a, j], R[b, j] = R[b, j], R[a, j]
        sols.append(R)
    return sols


insts = list(premkumar_instances())
insts += [scaled_standard(p, 9, 9) for p in PREMKUMAR_6x6 if p != 'none']
for s in (6, 9):
    for kind in ('rect', 'cluster', 'scatter'):
        for L in (2, 3, 4):
            insts.append(random_instance(s, s, kind, 0.3, levels=L, seed=500 + s))
for kind in ('rect', 'cluster', 'scatter'):
    insts.append(random_instance(6, 6, kind, 0.3, levels='continuous', seed=506))

out_p = open('results/sensitivity_power.csv', 'w', newline='')
out_c = open('results/rq3_spearman.csv', 'w', newline='')
wp, wc = csv.writer(out_p), csv.writer(out_c)
wp.writerow(['instance', 'm', 'levels', 'config', 'status', 'model', 'power_W'])
wc.writerow(['instance', 'm', 'levels', 'objective', 'spearman', 'p_best', 'p_ls_from_tct', 'p_ls_from_best_surrogate'])
rng = np.random.default_rng(7)
for inst in insts:
    t0 = time.time()
    lev = inst.meta.get('levels', 'std')
    res = 100.0 if inst.meta.get('source') in ('Premkumar2026', 'scaled_standard') or lev != 'continuous' else 1.0
    cfg = {'TCT': (inst.identity(), 'n/a')}
    for ob in ('var', 'L1', 'range', 'bot'):
        r = exact.solve_cpsat(inst, 'A', ob, time_limit=60, workers=1, resolution=res)
        cfg[f'opt_{ob}'] = (r['R'], r['status'])
    for name, (R, st) in cfg.items():
        for mk, mdl in models.items():
            wp.writerow([inst.name, inst.m, lev, name, st, mk, round(mdl.power(inst, R), 3)])
    # RQ3
    pw = {k: base.power(inst, R) for k, (R, _) in cfg.items()}
    best_sur = max((k for k in cfg if k != 'TCT'), key=lambda k: pw[k])
    _, p_tct = power_ls(inst, base, cfg['TCT'][0])
    _, p_bs = power_ls(inst, base, cfg[best_sur][0])
    p_best = max(p_tct, p_bs, *pw.values())
    sols = pool(inst, cfg['opt_var'][0], rng)
    P = np.array([base.power(inst, R) for R in sols])
    for ob, f in OBJECTIVES.items():
        V = np.array([f(inst.row_sums(R)) for R in sols])
        rho = spearmanr(V, -P).correlation        # +1 = lower surrogate <=> higher power
        wc.writerow([inst.name, inst.m, lev, ob, round(float(rho), 4), round(p_best, 3),
                     round(p_tct, 3), round(p_bs, 3)])
    out_p.flush(); out_c.flush()
    print(inst.name, round(time.time() - t0, 1), 's', flush=True)
