"""Rearrangement with regrouping (RRG) for variant B (free assignment of modules to rows).

Idea. Any variant-B solution, an m x n matrix of module indices, can be read as a
variant-A solution of a *virtual* matrix whose columns are the columns of the solution
matrix. Within that virtual matrix, RA and block rearrangement take exact,
monotone best-response steps (Boudt et al. 2018). The grouping of modules into virtual
columns is arbitrary: permuting the entries inside a row leaves every row total, and
hence the objective, unchanged, but changes which modules RA may exchange next.
RRG therefore alternates
  (1) rearrangement: RA / random-block rearrangement on the current virtual columns
      until no improvement, and
  (2) regrouping: a new assignment of each row's modules to virtual columns, which
      costs nothing in objective value.
Every accepted step is non-worsening, so the search never moves uphill; regrouping
provides the diversification that a perturbation provides in ILS without a loss.
"""
import numpy as np
from .heuristics import Budget, bldm, ra_single, _block_rearrange, ils
from .objectives import OBJECTIVES


def regroup(inst, R, rng, rule='mixed'):
    """Permute entries within each row. 'sorted': j-th largest of each row forms column j
    (columns of similar magnitude, so the next RA pairs large with small across rows);
    'random': independent random permutation of every row; 'mixed': sorted with
    probability 1/2, otherwise random."""
    flat = inst.flat; R = R.copy()
    if rule == 'mixed': rule = 'sorted' if rng.random() < 0.5 else 'random'
    for i in range(R.shape[0]):
        if rule == 'sorted':
            key = -flat[R[i]] + 1e-9 * rng.random(R.shape[1])       # random tie break
            R[i] = R[i][np.argsort(key, kind='stable')]
        else:
            R[i] = R[i][rng.permutation(R.shape[1])]
    return R


def _rearrange(inst, R, rng, budget, block_iters):
    """RA to convergence, then random-block rearrangement until `block_iters`
    consecutive non-improving blocks, then RA again."""
    flat = inst.flat; n = inst.n
    R = ra_single(inst, R, budget=budget)
    S = flat[R].sum(axis=1); cur = S.var(); stall = 0
    while stall < block_iters and budget.left:
        block = rng.random(n) < 0.5
        if block.all() or not block.any(): continue
        newR = _block_rearrange(inst, R, block); budget.spend()
        v = flat[newR].sum(axis=1).var()
        if v < cur - 1e-12: R, cur, stall = newR, v, 0
        else: stall += 1
    return ra_single(inst, R, budget=budget)


def rrg(inst, budget_evals=10_000, seed=0, R0=None, rule='mixed', block_iters=None, ils_frac=0.0, kick=4, trace=None, regrouping=True, start='bldm'):
    """Rearrangement with regrouping. Start: BLDM (deterministic) unless R0 is given.
    Each RA column step and each block step counts as one evaluation."""
    rng = np.random.default_rng(seed); flat = inst.flat
    own = int(round(budget_evals * (1 - ils_frac)))
    budget = Budget(own)
    block_iters = max(3, inst.n // 2) if block_iters is None else block_iters
    if R0 is not None: R = R0.copy()
    elif start == 'bldm': R = bldm(inst)
    else: R = rng.permutation(inst.m * inst.n).reshape(inst.m, inst.n)    # random balanced assignment
    budget.spend()
    R = _rearrange(inst, R, rng, budget, block_iters)
    best, bestv = R, flat[R].sum(axis=1).var()
    if trace is not None: trace.append((budget.used, bestv))
    while budget.left and regrouping:
        if bestv <= 1e-12: break
        Rg = regroup(inst, best, rng, rule)
        Rn = _rearrange(inst, Rg, rng, budget, block_iters)
        v = flat[Rn].sum(axis=1).var()
        if v <= bestv + 1e-12:                     # accept equal: plateau moves
            if v < bestv - 1e-12 and trace is not None: trace.append((budget.used, v))
            best, bestv = Rn, v
    if ils_frac > 0 and bestv > 1e-12:             # remaining budget: exchange-based ILS from the result
        best = ils(inst, 'B', 'var', budget_evals - min(budget.used, own), kick=kick, seed=seed, R0=best)
    return best
