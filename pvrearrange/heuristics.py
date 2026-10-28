"""Classical heuristics, described in standard optimisation terms.

A solution is the index matrix R (see instance.py). All heuristics count
"evaluations" = computations of the full row-sum vector, so that budgets are
comparable with population-based metaheuristics.
"""
from __future__ import annotations
import heapq
import numpy as np
from scipy.optimize import linear_sum_assignment
from .objectives import OBJECTIVES


class Budget:
    def __init__(self, max_evals=None):
        self.max = max_evals; self.used = 0
    def spend(self, k=1):
        self.used += k
    @property
    def left(self):
        return self.max is None or self.used < self.max


def _rng(seed):
    return np.random.default_rng(seed)


def random_solution(inst, variant, rng):
    m, n = inst.m, inst.n
    if variant == 'A':
        R = inst.identity().copy()
        for j in range(n): R[:, j] = R[rng.permutation(m), j]
        return R
    return rng.permutation(m * n).reshape(m, n)


# ---------------------------------------------------------------------------
# H1  Greedy column-by-column opposite ordering (Coffman & Yannakakis 1984)
# ---------------------------------------------------------------------------
def greedy_columns(inst):
    """Variant A. Columns in decreasing order of variance; each column's entries are
    placed so that larger values go to rows with smaller partial sums."""
    m, n = inst.m, inst.n
    G = inst.G
    order = np.argsort(-G.var(axis=0), kind='stable')
    R = np.empty((m, n), dtype=np.int64)
    partial = np.zeros(m)
    for j in order:
        ks = np.argsort(-G[:, j], kind='stable')          # largest entries first
        rows = np.argsort(partial, kind='stable')         # smallest partial sums first
        for k, i in zip(ks, rows):
            R[i, j] = k * n + j
        partial += inst.flat[R[:, j]]
    return R


# ---------------------------------------------------------------------------
# H2  Rearrangement Algorithm (Puccetti & Rueschendorf 2012; Embrechts et al. 2013)
# ---------------------------------------------------------------------------
def _oppositely_order(vals_idx, flat, rest):
    """Return the column (as module indices) arranged oppositely to `rest`."""
    v = flat[vals_idx]
    srt = vals_idx[np.argsort(-v, kind='stable')]         # values descending
    out = np.empty_like(vals_idx)
    out[np.argsort(rest, kind='stable')] = srt            # largest value -> smallest rest
    return out


def ra_single(inst, R0, eps=0.0, budget=None, max_sweeps=10_000):
    """One RA run from R0 (variant A). Sweeps over columns until a full sweep changes
    nothing or the variance improves by less than eps."""
    R = R0.copy(); flat = inst.flat; n = inst.n
    S = flat[R].sum(axis=1)
    prev = S.var()
    for _ in range(max_sweeps):
        changed = False
        for j in range(n):
            if budget is not None and not budget.left: return R
            rest = S - flat[R[:, j]]
            new = _oppositely_order(R[:, j], flat, rest)
            if not np.array_equal(flat[new], flat[R[:, j]]):
                R[:, j] = new; changed = True
            S = rest + flat[R[:, j]]
            if budget is not None: budget.spend()
        cur = S.var()
        if not changed or prev - cur <= eps: break
        prev = cur
    return R


def ra(inst, objective='var', restarts=30, seed=0, budget=None, start_identity=True, eps=0.0):
    """RA with random restarts; returns best solution under `objective`."""
    rng = _rng(seed); f = OBJECTIVES[objective]
    best, bestv = None, np.inf
    for r in range(restarts):
        if budget is not None and not budget.left: break
        R0 = inst.identity() if (r == 0 and start_identity) else random_solution(inst, 'A', rng)
        R = ra_single(inst, R0, eps, budget)
        v = f(inst.row_sums(R))
        if v < bestv: best, bestv = R, v
    return best


# ---------------------------------------------------------------------------
# H3  Block rearrangement: random blocks (BRA) and BRAVE (Boudt et al. 2018)
# ---------------------------------------------------------------------------
def _block_rearrange(inst, R, block):
    """Permute rows of the column block so that its partial row sums are oppositely
    ordered to the partial sums of the complementary block."""
    flat = inst.flat
    Sb = flat[R[:, block]].sum(axis=1)
    Sc = flat[R[:, ~block]].sum(axis=1)
    src = np.argsort(-Sb, kind='stable')                  # block rows, largest first
    dst = np.argsort(Sc, kind='stable')                   # complement rows, smallest first
    newR = R.copy()
    newR[dst[:, None], np.where(block)[0][None, :]] = R[src][:, block]
    return newR


def _partition_greedy(beta):
    """Two-way number partitioning of beta (greedy, largest |beta| first)."""
    d = np.zeros(len(beta), bool); s1 = s2 = 0.0
    for j in np.argsort(-np.abs(beta), kind='stable'):
        if s1 <= s2: d[j] = True; s1 += beta[j]
        else: s2 += beta[j]
    return d


def _partition_kk(beta):
    """Two-way Karmarkar-Karp differencing with back-tracking of the partition."""
    heap = [(-b, j, ((j,), ())) for j, b in enumerate(beta)]
    heapq.heapify(heap)
    while len(heap) > 1:
        a, ja, (a1, a2) = heapq.heappop(heap)
        b, jb, (b1, b2) = heapq.heappop(heap)
        heapq.heappush(heap, (a - b, ja, (a1 + b2, a2 + b1)))
    _, _, (s1, _) = heap[0]
    d = np.zeros(len(beta), bool); d[list(s1)] = True
    return d


def block_ra(inst, R0=None, iters=200, rule='brave', partition='greedy', seed=0,
             then_ra=True, brave_iters=30, budget=None):
    """BRA (rule='random') or BRAVE (rule='brave'); BRAVE+RA switches to RA after
    `brave_iters` iterations or when no improvement is observed (Boudt et al. 2018)."""
    rng = _rng(seed); n = inst.n; flat = inst.flat
    R = inst.identity() if R0 is None else R0.copy()
    S = flat[R].sum(axis=1); cur = S.var()
    part = _partition_kk if partition == 'kk' else _partition_greedy
    stall = 0
    for it in range(iters):
        if budget is not None and not budget.left: break
        if rule == 'brave' and stall == 0:
            X = flat[R]
            beta = ((X - X.mean(axis=0)) * (S - S.mean())[:, None]).mean(axis=0)
            block = part(beta)
        else:
            block = rng.random(n) < 0.5
        if block.all() or not block.any():
            block = np.zeros(n, bool); block[rng.integers(n)] = True
        newR = _block_rearrange(inst, R, block)
        if budget is not None: budget.spend()
        S2 = flat[newR].sum(axis=1); v = S2.var()
        if v < cur - 1e-12:
            R, S, cur, stall = newR, S2, v, 0
        else:
            stall += 1                      # next iteration uses a random block
        if then_ra and rule == 'brave' and (it + 1 >= brave_iters or stall >= 3):
            break                           # BRAVE as pre-solver, then RA
    if then_ra:
        R = ra_single(inst, R, budget=budget)
    return R


# ---------------------------------------------------------------------------
# H4  Balanced Largest Differencing Method (variant B)
# ---------------------------------------------------------------------------
def bldm(inst):
    """Balanced m-way differencing: sorted weights are cut into n groups of m (one
    module per row); groups are merged pairwise, largest-range first, pairing the
    largest partial sum of one with the smallest of the other."""
    m, n = inst.m, inst.n; flat = inst.flat
    order = np.argsort(-flat, kind='stable')
    heap = []
    for t in range(n):
        idx = order[t * m:(t + 1) * m]
        sums = flat[idx].astype(float)
        members = [[int(p)] for p in idx]
        heapq.heappush(heap, (-(sums.max() - sums.min()), t, sums, members))
    tie = n
    while len(heap) > 1:
        _, _, sa, ma = heapq.heappop(heap)
        _, _, sb, mb = heapq.heappop(heap)
        ia = np.argsort(-sa, kind='stable'); ib = np.argsort(sb, kind='stable')
        sums = sa[ia] + sb[ib]
        members = [ma[a] + mb[b] for a, b in zip(ia, ib)]
        heapq.heappush(heap, (-(sums.max() - sums.min()), tie, sums, members)); tie += 1
    _, _, _, members = heap[0]
    return np.asarray(members, dtype=np.int64)


# ---------------------------------------------------------------------------
# H5  PV-specific heuristics (variant B)
# ---------------------------------------------------------------------------
def bwsa(inst):
    """Best-worst sorting algorithm (Storey et al. 2013). Rows of a sorted (n x m) matrix
    are paired, the second one reversed, and added; repeated until one row of m tiers is
    left. A zero row is added when the number of rows is odd."""
    m, n = inst.m, inst.n; flat = inst.flat
    order = list(np.argsort(-flat, kind='stable'))
    rows = []
    for t in range(n):
        idx = order[t * m:(t + 1) * m]
        rows.append([(float(flat[p]), [int(p)]) for p in idx])
    while len(rows) > 1:
        rows = [sorted(r, key=lambda e: -e[0]) for r in rows]
        rows.sort(key=lambda r: -r[0][0])
        if len(rows) % 2: rows.append([(0.0, []) for _ in range(m)])
        new = []
        for a, b in zip(rows[0::2], rows[1::2]):
            new.append([(ea[0] + eb[0], ea[1] + eb[1]) for ea, eb in zip(a, b[::-1])])
        rows = new
    return np.asarray([e[1] for e in rows[0]], dtype=np.int64)


def subset_sum_rows(inst, resolution=10.0):
    """Riva Sanseverino et al. (2015), adapted to rows of exactly n modules: rows are built
    one at a time; each row contains the largest remaining module and the other n-1
    modules are chosen by dynamic programming so that the row sum is closest to the
    average row sum. Suitable for small and medium arrays."""
    m, n = inst.m, inst.n; flat = inst.flat
    w = np.rint(flat / resolution).astype(int)
    target = w.sum() / m
    remaining = list(np.argsort(-flat, kind='stable'))
    rows = []
    for _ in range(m - 1):
        first = remaining[0]; rest = remaining[1:]
        need = n - 1; tgt = target - w[first]
        # reach[c] : dict sum -> (prev_sum, item) for subsets of size c
        reach = [dict() for _ in range(need + 1)]; reach[0][0] = None
        for it, p in enumerate(rest):
            wp = w[p]
            for c in range(min(need, it + 1), 0, -1):
                for s in list(reach[c - 1].keys()):
                    ns = s + wp
                    if ns not in reach[c]: reach[c][ns] = (s, it)
        best = min(reach[need].keys(), key=lambda s: abs(s - tgt))
        chosen = []; c = need; s = best
        while c > 0:
            ps, it = reach[c][s]; chosen.append(rest[it]); s = ps; c -= 1
        row = [first] + chosen
        rows.append(row)
        rs = set(row); remaining = [p for p in remaining if p not in rs]
    rows.append(remaining)
    return np.asarray(rows, dtype=np.int64)


# ---------------------------------------------------------------------------
# H6  Swap local search + iterated local search
# ---------------------------------------------------------------------------
def local_search(inst, R0, variant, objective='var', budget=None, rng=None, first_improvement=True):
    """Pairwise swap neighbourhood. Variant A swaps two rows within one column; variant B
    swaps any two modules in different rows. Delta-evaluation on row sums."""
    f = OBJECTIVES[objective]; flat = inst.flat; m, n = inst.m, inst.n
    rng = rng or _rng(0)
    R = R0.copy(); S = flat[R].sum(axis=1); cur = f(S)
    improved = True
    while improved:
        improved = False
        if variant == 'A':
            moves = [(j, a, b) for j in rng.permutation(n) for a in range(m) for b in range(a + 1, m)]
        else:
            cells = [(i, j) for i in range(m) for j in range(n)]
            moves = None
        if variant == 'A':
            for j, a, b in moves:
                if budget is not None and not budget.left: return R
                pa, pb = R[a, j], R[b, j]; d = flat[pb] - flat[pa]
                if d == 0: continue
                S[a] += d; S[b] -= d; v = f(S)
                if budget is not None: budget.spend()
                if v < cur - 1e-12:
                    R[a, j], R[b, j] = pb, pa; cur = v; improved = True
                    if first_improvement: break
                else:
                    S[a] -= d; S[b] += d
        else:
            order = rng.permutation(len(cells))
            done = False
            for x in order:
                if done: break
                ia, ja = cells[x]
                for ib in range(m):
                    if ib == ia: continue
                    for jb in range(n):
                        if budget is not None and not budget.left: return R
                        pa, pb = R[ia, ja], R[ib, jb]; d = flat[pb] - flat[pa]
                        if d == 0: continue
                        S[ia] += d; S[ib] -= d; v = f(S)
                        if budget is not None: budget.spend()
                        if v < cur - 1e-12:
                            R[ia, ja], R[ib, jb] = pb, pa; cur = v; improved = True
                            if first_improvement: done = True; break
                        else:
                            S[ia] -= d; S[ib] += d
                    if done: break
    return R


def ils(inst, variant, objective='var', budget_evals=10_000, kick=3, seed=0, R0=None):
    """Iterated local search: local search, then `kick` random swaps, accept if better."""
    rng = _rng(seed); f = OBJECTIVES[objective]; m, n = inst.m, inst.n
    B = Budget(budget_evals)
    R = local_search(inst, inst.identity() if R0 is None else R0, variant, objective, B, rng)
    best, bestv = R, f(inst.row_sums(R))
    while B.left:
        X = best.copy()
        for _ in range(kick):
            if variant == 'A':
                j = rng.integers(n); a, b = rng.choice(m, 2, replace=False)
                X[a, j], X[b, j] = X[b, j], X[a, j]
            else:
                ia, ib = rng.choice(m, 2, replace=False); ja, jb = rng.integers(n), rng.integers(n)
                X[ia, ja], X[ib, jb] = X[ib, jb], X[ia, ja]
        X = local_search(inst, X, variant, objective, B, rng)
        v = f(inst.row_sums(X))
        if v < bestv - 1e-12: best, bestv = X, v
    return best


# ---------------------------------------------------------------------------
# Post-processing shared by all methods: relabel rows to minimise module moves
# ---------------------------------------------------------------------------
def min_moves_relabel(R_prev, R_new):
    """Rows of a TCT array are interchangeable. Relabel the rows of R_new so that the
    number of modules that change row with respect to R_prev is minimal (Munkres), as in
    Riva Sanseverino et al. (2015). Returns (R_relabelled, moves)."""
    m = R_prev.shape[0]
    prev_sets = [set(r.tolist()) for r in R_prev]
    new_sets = [set(r.tolist()) for r in R_new]
    C = np.array([[len(new_sets[b] - prev_sets[a]) for b in range(m)] for a in range(m)])
    ra_, cb = linear_sum_assignment(C)
    Rr = R_new[cb]
    return Rr, int(C[ra_, cb].sum())
