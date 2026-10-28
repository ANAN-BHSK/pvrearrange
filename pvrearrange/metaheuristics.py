"""Population-based metaheuristics from the PV reconfiguration literature (M1-M5), with
random-key decoding (Bean 1994), described in metaphor-free terms.

Every method searches a continuous key vector x in [0,1]^(m*n):
  variant B: argsort(x) is a permutation of all modules, cut into m rows of n modules;
  variant A: keys are reshaped to m x n and each column is argsorted separately, giving
             the row order of the modules of that column.
All methods share one evaluation budget (number of decoded solutions evaluated) and
return (best R, best value, history of best values per evaluation block).
"""
from __future__ import annotations
import math
import numpy as np


# ---------------------------------------------------------------------------
# Vectorised decoding and evaluation
# ---------------------------------------------------------------------------
def decode(inst, X, variant):
    """X: (N, m*n) keys -> R: (N, m, n) index matrices."""
    N = X.shape[0]; m, n = inst.m, inst.n
    if variant == 'B':
        return np.argsort(X, axis=1, kind='stable').reshape(N, m, n)
    perm = np.argsort(X.reshape(N, m, n), axis=1, kind='stable')     # row order per column
    return perm * n + np.arange(n)[None, None, :]


def _obj_vec(S, objective):
    if objective == 'var':   return S.var(axis=1)
    if objective == 'L1':    return np.abs(S - S.mean(axis=1, keepdims=True)).sum(axis=1)
    if objective == 'range': return S.max(axis=1) - S.min(axis=1)
    if objective == 'bot':   return -S.min(axis=1)
    raise ValueError(objective)


class Evaluator:
    def __init__(self, inst, variant, objective, budget):
        self.inst, self.variant, self.objective = inst, variant, objective
        self.budget, self.used = budget, 0
        self.best_val, self.best_R = np.inf, None
        self.trace = []                                   # (evals used, best value)

    @property
    def left(self): return self.budget - self.used

    def __call__(self, X):
        X = np.atleast_2d(X)
        k = min(len(X), self.left)
        if k <= 0: return np.full(len(X), np.inf)
        R = decode(self.inst, X[:k], self.variant)
        S = self.inst.flat[R].sum(axis=2)
        f = _obj_vec(S, self.objective)
        self.used += k
        i = int(np.argmin(f))
        if f[i] < self.best_val:
            self.best_val, self.best_R = float(f[i]), R[i].copy()
        self.trace.append((self.used, self.best_val))
        out = np.full(len(X), np.inf); out[:k] = f
        return out


def _result(ev):
    return dict(R=ev.best_R, value=ev.best_val, evals=ev.used, trace=ev.trace)


# ---------------------------------------------------------------------------
# M0  Random search with the same decoding (ablation baseline)
# ---------------------------------------------------------------------------
def random_search(inst, variant='B', objective='L1', budget=10_000, pop=40, seed=0):
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    while ev.left > 0:
        ev(rng.random((pop, D)))
    return _result(ev)


# ---------------------------------------------------------------------------
# M1  Random-key genetic algorithm with elitism (biased random-key GA, Bean 1994 style)
# ---------------------------------------------------------------------------
def rkga(inst, variant='B', objective='L1', budget=10_000, pop=40, elite=0.2, mutants=0.1,
         bias=0.7, seed=0):
    """Each generation keeps the best `elite` fraction, adds `mutants` fresh random key
    vectors, and fills the rest by uniform crossover between an elite and a non-elite
    parent, taking each key from the elite parent with probability `bias`."""
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    X = rng.random((pop, D)); f = ev(X)
    ne, nm = max(1, int(elite * pop)), max(1, int(mutants * pop))
    while ev.left > 0:
        o = np.argsort(f); X, f = X[o], f[o]
        E, NE = X[:ne], X[ne:]
        nc = pop - ne - nm
        a = E[rng.integers(ne, size=nc)]; b = NE[rng.integers(len(NE), size=nc)]
        child = np.where(rng.random((nc, D)) < bias, a, b)
        new = np.vstack([child, rng.random((nm, D))])
        fn = ev(new)
        X = np.vstack([E, new]); f = np.concatenate([f[:ne], fn])
    return _result(ev)


# ---------------------------------------------------------------------------
# M2  Particle swarm optimisation (global best, linearly decreasing inertia)
# ---------------------------------------------------------------------------
def pso(inst, variant='B', objective='L1', budget=10_000, pop=40, w=(0.9, 0.4), c1=2.0, c2=2.0,
        vmax=0.2, seed=0):
    """Velocity = inertia x velocity + c1 r1 (personal best - x) + c2 r2 (global best - x)."""
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    X = rng.random((pop, D)); V = np.zeros((pop, D)); f = ev(X)
    P, fp = X.copy(), f.copy()
    iters = max(1, budget // pop - 1); t = 0
    while ev.left > 0:
        g = P[np.argmin(fp)]
        wt = w[0] - (w[0] - w[1]) * min(1.0, t / iters)
        V = wt * V + c1 * rng.random((pop, D)) * (P - X) + c2 * rng.random((pop, D)) * (g - X)
        V = np.clip(V, -vmax, vmax)
        X = np.clip(X + V, 0, 1)
        f = ev(X)
        imp = f < fp; P[imp], fp[imp] = X[imp], f[imp]; t += 1
    return _result(ev)


# ---------------------------------------------------------------------------
# M3  Differential evolution, DE/rand/1/bin
# ---------------------------------------------------------------------------
def de(inst, variant='B', objective='L1', budget=10_000, pop=40, F=0.5, CR=0.9, seed=0):
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    X = rng.random((pop, D)); f = ev(X)
    idx = np.arange(pop)
    while ev.left > 0:
        r = np.array([rng.choice(np.delete(idx, i), 3, replace=False) for i in idx])
        Vm = X[r[:, 0]] + F * (X[r[:, 1]] - X[r[:, 2]])
        cross = rng.random((pop, D)) < CR
        cross[idx, rng.integers(D, size=pop)] = True
        U = np.clip(np.where(cross, Vm, X), 0, 1)
        fu = ev(U)
        better = fu <= f; X[better], f[better] = U[better], fu[better]
    return _result(ev)


# ---------------------------------------------------------------------------
# M4  GOA (Saremi et al. 2017) as used by Fathy (2018)
# ---------------------------------------------------------------------------
def goa(inst, variant='B', objective='L1', budget=10_000, pop=40, cmax=1.0, cmin=1e-5, fs=0.5, ls=1.5,
        seed=0):
    """Metaphor-free description: each solution moves to the best-so-far solution plus a
    weighted sum of pairwise attraction/repulsion terms s(d) = f e^(-d/l) - e^(-d) applied
    to normalised pairwise distances; the step coefficient c decreases linearly, which
    shrinks the move from exploration to convergence on the best solution."""
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    X = rng.random((pop, D)); f = ev(X)
    best = X[np.argmin(f)].copy(); fb = f.min()
    iters = max(1, budget // pop - 1); t = 0
    s = lambda r: fs * np.exp(-r / ls) - np.exp(-r)
    while ev.left > 0:
        c = cmax - t * (cmax - cmin) / iters
        diff = X[None, :, :] - X[:, None, :]                       # x_j - x_i
        dist = np.linalg.norm(diff, axis=2) + 1e-12
        dn = 2 + np.remainder(dist, 2)                              # distance mapped to [2,4)
        coef = c * 0.5 * s(dn) / dist                               # (ub-lb)/2 = 0.5
        np.fill_diagonal(coef, 0.0)
        Si = (coef[:, :, None] * diff).sum(axis=1)
        X = np.clip(c * Si + best, 0, 1)
        f = ev(X)
        i = int(np.argmin(f))
        if f[i] < fb: fb, best = f[i], X[i].copy()
        t += 1
    return _result(ev)


# ---------------------------------------------------------------------------
# M5  HHO (Heidari et al. 2019) as used by Yousri et al. (2020)
# ---------------------------------------------------------------------------
def _levy(rng, size, beta=1.5):
    sig = (math.gamma(1 + beta) * math.sin(math.pi * beta / 2) /
           (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return 0.01 * rng.normal(0, sig, size) / np.abs(rng.normal(0, 1, size)) ** (1 / beta)


def hho(inst, variant='B', objective='L1', budget=10_000, pop=40, seed=0):
    """Metaphor-free description: an 'escape energy' E = 2 E0 (1 - t/T) controls the phase.
    |E| >= 1: global moves (towards a random solution or around the population mean).
    |E| < 1: moves towards the best solution, either directly (scaled by E) or via a
    trial step plus a heavy-tailed (Levy) perturbation that is accepted only if it
    improves; trial steps cost extra evaluations."""
    rng = np.random.default_rng(seed); D = inst.m * inst.n
    ev = Evaluator(inst, variant, objective, budget)
    X = rng.random((pop, D)); f = ev(X)
    b = int(np.argmin(f)); best = X[b].copy(); fbest = f[b]          # best-so-far ("rabbit")
    iters = max(1, budget // (pop * 2)); t = 0
    while ev.left > 0:
        E0 = 2 * rng.random(pop) - 1
        E = 2 * E0 * (1 - min(t, iters) / iters)
        Xn = X.copy()
        for i in range(pop):
            q, r = rng.random(), rng.random()
            if abs(E[i]) >= 1:
                if q >= 0.5:
                    Xr = X[rng.integers(pop)]
                    Xn[i] = Xr - rng.random() * np.abs(Xr - 2 * rng.random() * X[i])
                else:
                    Xn[i] = (best - X.mean(axis=0)) - rng.random() * rng.random(D)
            else:
                J = 2 * (1 - rng.random())
                if r >= 0.5 and abs(E[i]) >= 0.5:
                    Xn[i] = (best - X[i]) - E[i] * np.abs(J * best - X[i])
                elif r >= 0.5:
                    Xn[i] = best - E[i] * np.abs(best - X[i])
                else:
                    ref = X[i] if abs(E[i]) >= 0.5 else X.mean(axis=0)
                    Y = np.clip(best - E[i] * np.abs(J * best - ref), 0, 1)
                    Z = np.clip(Y + rng.random(D) * _levy(rng, D), 0, 1)
                    fy, fz = ev(np.vstack([Y, Z]))
                    if fy < f[i] and fy <= fz: Xn[i] = Y
                    elif fz < f[i]: Xn[i] = Z
        X = np.clip(Xn, 0, 1)                                        # positions replaced, as in HHO
        f = ev(X)
        i = int(np.argmin(f))
        if f[i] < fbest: fbest, best = f[i], X[i].copy()
        t += 1
    return _result(ev)


METAHEURISTICS = {'RandomSearch': random_search, 'RKGA': rkga, 'PSO': pso, 'DE': de, 'GOA': goa, 'HHO': hho}
