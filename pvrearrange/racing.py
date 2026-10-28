"""Iterated racing for algorithm configuration, following the irace procedure
(Lopez-Ibanez et al. 2016, Operations Research Perspectives 3:43-58).

R/irace could not be installed in the computing environment, so this module
re-implements its main loop:
  1. sample N_j configurations (uniformly in iteration 1, around elites afterwards);
  2. race them on a stream of training instances: after T_first instances, apply a
     Friedman test after every instance and discard configurations whose rank sum is
     significantly worse than the best (Conover post-hoc, alpha = 0.05);
  3. keep up to N_min survivors as elites; numeric sampling widths shrink as
     (1 / N_new)^(1/d) per iteration; categorical probabilities shift towards the elite.
Results of elites on instances already seen are reused.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon, t as tdist, rankdata


class Param:
    def __init__(self, kind, lo=None, hi=None, values=None, log=False):
        self.kind, self.lo, self.hi, self.values, self.log = kind, lo, hi, values, log

    def _to(self, x):   return math.log(x) if self.log else x
    def _from(self, y): return math.exp(y) if self.log else y

    def uniform(self, rng):
        if self.kind == 'cat': return self.values[rng.integers(len(self.values))]
        y = rng.uniform(self._to(self.lo), self._to(self.hi))
        v = self._from(y)
        return int(round(v)) if self.kind == 'int' else float(v)

    def around(self, rng, x, sd_frac, it, n_iter):
        if self.kind == 'cat':
            keep = 1 - (it - 1) / max(1, n_iter)              # irace: prob. shifts to elite value
            return x if rng.random() > keep * (len(self.values) - 1) / len(self.values) else self.uniform(rng)
        a, b = self._to(self.lo), self._to(self.hi)
        for _ in range(100):                                    # truncated normal
            y = rng.normal(self._to(x), sd_frac * (b - a))
            if a <= y <= b: break
        else:
            y = min(max(y, a), b)
        v = self._from(y)
        return int(round(v)) if self.kind == 'int' else float(v)


def _conover_survivors(Y, alpha=0.05):
    """Y: (blocks x configs) costs. Returns boolean mask of configs not significantly
    worse than the best, or None if the Friedman test is not significant."""
    b, k = Y.shape
    if k < 2: return np.ones(k, bool)
    R = np.vstack([rankdata(row) for row in Y])
    if np.all(R == R[0]): return None
    if k == 2:                                                   # two survivors: Wilcoxon signed-rank
        dlt = Y[:, 0] - Y[:, 1]
        if np.allclose(dlt, 0): return None
        try:
            p = wilcoxon(Y[:, 0], Y[:, 1], zero_method='zsplit').pvalue
        except ValueError:
            return None
        if not (p < alpha): return None
        return np.array([dlt.mean() <= 0, dlt.mean() >= 0])
    try:
        p = friedmanchisquare(*Y.T).pvalue
    except ValueError:
        return None
    if not (p < alpha): return None
    Rj = R.sum(axis=0); A = (R ** 2).sum()
    denom = (b - 1) * (k - 1)
    var = 2 * (b * A - (Rj ** 2).sum()) / denom
    if var <= 0: return None
    cd = tdist.ppf(1 - alpha / 2, denom) * math.sqrt(var)
    return Rj - Rj.min() <= cd


def irace(space, run, instances, budget, seed=0, t_first=5, alpha=0.05, log=print, default=None):
    """space: dict name -> Param. run(config, instance) -> cost (lower is better).
    instances: list (stream is cycled in a random order). Returns (best config, history)."""
    rng = np.random.default_rng(seed)
    d = len(space)
    n_iter = int(2 + math.log2(d))
    n_min = int(2 + math.log2(d))
    order = list(rng.permutation(len(instances)))
    cache = {}                                                   # (cfg_id, inst_idx) -> cost
    cfgs = {}                                                    # cfg_id -> dict
    used = 0; next_id = 0; elites = []
    history = []

    def evaluate(cid, ii):
        nonlocal used
        key = (cid, ii)
        if key not in cache:
            cache[key] = float(run(cfgs[cid], instances[ii])); used += 1
        return cache[key]

    sd_frac = 0.5
    for it in range(1, n_iter + 1):
        left = budget - used
        if left <= 0: break
        Bj = left / (n_iter - it + 1)
        Nj = max(n_min + 1, int(Bj / (t_first + min(5, it))))
        new = []
        n_new = max(1, Nj - len(elites))
        if it > 1: sd_frac *= (1.0 / n_new) ** (1.0 / d)
        for q in range(n_new):
            if it == 1:
                c = default if (q == 0 and default) else {k: p.uniform(rng) for k, p in space.items()}
            else:
                w = np.arange(len(elites), 0, -1, dtype=float); w /= w.sum()
                parent = cfgs[elites[rng.choice(len(elites), p=w)]]
                c = {k: p.around(rng, parent[k], sd_frac, it, n_iter) for k, p in space.items()}
            cfgs[next_id] = dict(c); new.append(next_id); next_id += 1
        alive = list(elites) + new
        seen = []
        # race
        stream = order[:] if it == 1 else list(rng.permutation(len(instances)))
        iter_start = used
        for ii in stream:
            if used - iter_start >= Bj or len(alive) <= n_min and len(seen) >= t_first: break
            for cid in alive: evaluate(cid, ii)
            seen.append(ii)
            if len(seen) >= t_first and len(alive) > 1:
                Y = np.array([[cache[(cid, s)] for cid in alive] for s in seen])
                keep = _conover_survivors(Y, alpha)
                if keep is not None:
                    alive = [cid for cid, kk in zip(alive, keep) if kk]
        Y = np.array([[cache[(cid, s)] for cid in alive] for s in seen])
        mr = np.vstack([rankdata(r) for r in Y]).mean(axis=0)
        elites = [alive[i] for i in np.argsort(mr)][:n_min]
        history.append(dict(iteration=it, sampled=len(new), survivors=len(alive), used=used,
                            best=dict(cfgs[elites[0]])))
        log(f'  it {it}: sampled {len(new)}, survivors {len(alive)}, budget used {used}, best {cfgs[elites[0]]}')
    return cfgs[elites[0]], history
