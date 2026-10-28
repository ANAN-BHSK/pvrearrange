"""Exact methods.

E1: CP-SAT models for variant A (within-column) and B (free balanced assignment).
    Modules with equal weight are aggregated (count variables), which removes the
    symmetry between identical modules; rows are ordered by decreasing row sum,
    which removes the symmetry between interchangeable TCT rows.
E4: closed-form optimum for two-valued matrices (Haus 2014, Theorem 1; Bellini et al.
    2016, Theorem 2): balancing the number of high entries per row is optimal for every
    Schur-convex objective.
"""
from __future__ import annotations
import time
from collections import Counter
import numpy as np
from ortools.sat.python import cp_model
from .objectives import OBJECTIVES


def _scale(G, resolution):
    g = np.rint(G / resolution).astype(np.int64)
    return g


def solve_cpsat(inst, variant='A', objective='var', time_limit=60.0, workers=1,
                resolution=1.0, hint=None):
    """Return dict(status, obj, bound, time, R). obj/bound are in original units of the
    *rounded* instance (weights rounded to `resolution`)."""
    m, n = inst.m, inst.n
    g = _scale(inst.G, resolution)
    T = int(g.sum())
    md = cp_model.CpModel()
    ub_row = int(g.max()) * n

    if variant == 'A':
        # c[j][v][i] in {0,1}: one entry of value v from column j goes to row i
        cols = []
        S_terms = [[] for _ in range(m)]
        for j in range(n):
            cnt = Counter(g[:, j].tolist())
            cv = {}
            for v, c in cnt.items():
                xs = [md.NewIntVar(0, min(1, c), '') for _ in range(m)]
                md.Add(sum(xs) == c)
                cv[v] = xs
                for i in range(m): S_terms[i].append(v * xs[i])
            for i in range(m):
                md.Add(sum(cv[v][i] for v in cv) == 1)
            cols.append(cv)
        model_vars = ('A', cols)
    elif variant == 'B':
        cnt = Counter(g.reshape(-1).tolist())
        cv = {}
        S_terms = [[] for _ in range(m)]
        for v, c in cnt.items():
            xs = [md.NewIntVar(0, min(n, c), '') for _ in range(m)]
            md.Add(sum(xs) == c)
            cv[v] = xs
            for i in range(m): S_terms[i].append(v * xs[i])
        for i in range(m):
            md.Add(sum(cv[v][i] for v in cv) == n)
        model_vars = ('B', cv)
    else:
        raise ValueError(variant)

    S = [md.NewIntVar(0, ub_row, f'S{i}') for i in range(m)]
    for i in range(m): md.Add(S[i] == sum(S_terms[i]))
    for i in range(m - 1): md.Add(S[i] >= S[i + 1])      # row-symmetry breaking

    if objective == 'L1':
        d = [md.NewIntVar(0, m * ub_row + T, '') for _ in range(m)]
        for i in range(m): md.AddAbsEquality(d[i], m * S[i] - T)
        md.Minimize(sum(d))
    elif objective == 'range':
        md.Minimize(S[0] - S[m - 1])
    elif objective == 'bot':
        md.Maximize(S[m - 1])
    elif objective == 'var':
        sq = [md.NewIntVar(0, ub_row * ub_row, '') for _ in range(m)]
        for i in range(m): md.AddMultiplicationEquality(sq[i], [S[i], S[i]])
        md.Minimize(sum(sq))
    else:
        raise ValueError(objective)

    if hint is not None:   # complete warm start: assignment counts and row sums of a heuristic solution
        gf = g.reshape(-1)
        Hs = gf[np.asarray(hint)]                                   # m x n scaled values per row
        order = np.argsort(-Hs.sum(axis=1), kind='stable')          # respect S[0] >= S[1] >= ...
        Hs = Hs[order]; Hidx = np.asarray(hint)[order]
        for i in range(m): md.AddHint(S[i], int(Hs[i].sum()))
        if variant == 'A':
            for j, cv in enumerate(model_vars[1]):
                for v, xs in cv.items():
                    for i in range(m): md.AddHint(xs[i], int(gf[Hidx[i, j]] == v))
        else:
            for v, xs in model_vars[1].items():
                for i in range(m): md.AddHint(xs[i], int((Hs[i] == v).sum()))

    sv = cp_model.CpSolver()
    sv.parameters.max_time_in_seconds = float(time_limit)
    sv.parameters.num_workers = int(workers)
    t0 = time.perf_counter(); st = sv.Solve(md); t = time.perf_counter() - t0
    status = sv.StatusName(st)
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return dict(status=status, obj=None, bound=None, time=t, R=None)

    R = _reconstruct(inst, g, model_vars, sv, m, n)
    Sx = np.rint(inst.row_sums(R) / resolution) * resolution   # rounded instance
    obj = OBJECTIVES[objective](Sx)
    b = sv.BestObjectiveBound() if st == cp_model.FEASIBLE else sv.ObjectiveValue()
    bound = _bound_to_units(objective, b, T, m, resolution)
    return dict(status=status, obj=obj, bound=bound, time=t, R=R)


def _bound_to_units(objective, b, T, m, r):
    if objective == 'L1':    return b / m * r
    if objective == 'range': return b * r
    if objective == 'bot':   return -b * r
    if objective == 'var':   return (b / m - (T / m) ** 2) * r * r


def _reconstruct(inst, g, model_vars, sv, m, n):
    kind, data = model_vars
    R = -np.ones((m, n), dtype=np.int64)
    if kind == 'A':
        for j, cv in enumerate(data):
            pool = {}
            for k in range(m): pool.setdefault(int(g[k, j]), []).append(k * n + j)
            for v, xs in cv.items():
                for i in range(m):
                    if sv.Value(xs[i]): R[i, j] = pool[v].pop()
    else:
        pool = {}
        for p, v in enumerate(g.reshape(-1).tolist()): pool.setdefault(v, []).append(p)
        fill = [0] * m
        for v, xs in data.items():
            for i in range(m):
                for _ in range(sv.Value(xs[i])):
                    R[i, fill[i]] = pool[v].pop(); fill[i] += 1
    assert (R >= 0).all()
    return R


def two_valued_optimum(inst, variant='A'):
    """E4: optimal arrangement when the matrix contains exactly two distinct values.
    Distributes the high entries so that row counts differ by at most one, which is
    optimal for every Schur-convex objective. Returns None if not two-valued."""
    vals = np.unique(inst.G)
    if len(vals) != 2:
        return None
    hi = vals[1]
    m, n = inst.m, inst.n
    R = np.empty((m, n), dtype=np.int64)
    count = np.zeros(m, int)
    if variant == 'A':
        for j in range(n):
            col = inst.G[:, j]
            his = [k * n + j for k in range(m) if col[k] == hi]
            los = [k * n + j for k in range(m) if col[k] != hi]
            rows = np.argsort(count, kind='stable')            # fewest high entries first
            for t, i in enumerate(rows):
                if t < len(his): R[i, j] = his[t]; count[i] += 1
                else: R[i, j] = los[t - len(his)]
    else:
        flat = inst.flat
        his = [p for p in range(m * n) if flat[p] == hi]
        los = [p for p in range(m * n) if flat[p] != hi]
        q, r = divmod(len(his), m)
        it_h, it_l = iter(his), iter(los)
        for i in range(m):
            k = q + (1 if i < r else 0)
            R[i] = [next(it_h) for _ in range(k)] + [next(it_l) for _ in range(n - k)]
    return R
