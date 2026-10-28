"""Correctness tests. Run: python -m pytest -q tests"""
import itertools
import numpy as np
import pytest
from pvrearrange import premkumar_instances, random_instance, scaled_standard, OBJECTIVES, exact, heuristics as H
from pvrearrange.instance import Instance

P26 = {i.meta['pattern']: i for i in premkumar_instances()}

# L1 optima for variant A found in the pilot study (W/m2)
PILOT_L1 = {'none': 0, 'uneven_row': 0, 'uneven_column': 0, 'diagonal': 1133.33, 'short_narrow': 1333.33,
            'short_wide': 0, 'long_narrow': 266.67, 'long_wide': 0, 'central': 733.33}


def brute_force_A(inst, objective):
    """Enumerate all within-column arrangements (first column fixed)."""
    m, n = inst.m, inst.n; f = OBJECTIVES[objective]; best = np.inf
    cols = [list(itertools.permutations(range(m))) for _ in range(n - 1)]
    base = inst.identity()
    for ps in itertools.product(*cols):
        R = base.copy()
        for j, p in enumerate(ps, start=1): R[:, j] = base[list(p), j]
        best = min(best, f(inst.row_sums(R)))
    return best


def brute_force_B(inst, objective):
    m, n = inst.m, inst.n; f = OBJECTIVES[objective]; best = np.inf
    idx = list(range(m * n))
    def rec(rem, rows):
        nonlocal best
        if not rem:
            best = min(best, f(inst.row_sums(np.asarray(rows)))); return
        first = rem[0]
        for comb in itertools.combinations(rem[1:], n - 1):
            row = (first,) + comb
            rec([p for p in rem if p not in row], rows + [list(row)])
    rec(idx, [])
    return best


@pytest.mark.parametrize('pattern', list(PILOT_L1))
def test_cpsat_reproduces_pilot(pattern):
    r = exact.solve_cpsat(P26[pattern], 'A', 'L1', time_limit=60, workers=2, resolution=100)
    assert r['status'] == 'OPTIMAL'
    assert abs(r['obj'] - PILOT_L1[pattern]) < 0.5
    assert P26[pattern].is_feasible(r['R'], 'A')


@pytest.mark.parametrize('seed', range(4))
@pytest.mark.parametrize('objective', ['var', 'L1', 'range', 'bot'])
def test_cpsat_matches_brute_force(seed, objective):
    inst = random_instance(3, 4, 'scatter', 0.4, levels=3, seed=seed)
    rA = exact.solve_cpsat(inst, 'A', objective, workers=2)
    assert rA['status'] == 'OPTIMAL'
    assert abs(rA['obj'] - brute_force_A(inst, objective)) < 1e-6
    inst2 = random_instance(3, 3, 'scatter', 0.5, levels=3, seed=seed)
    rB = exact.solve_cpsat(inst2, 'B', objective, workers=2)
    assert abs(rB['obj'] - brute_force_B(inst2, objective)) < 1e-6
    assert inst2.is_feasible(rB['R'], 'B')


@pytest.mark.parametrize('seed', range(5))
def test_two_valued_is_optimal(seed):
    inst = random_instance(5, 6, 'rect', 0.3, levels=1, seed=seed)
    for variant in 'AB':
        R = exact.two_valued_optimum(inst, variant)
        assert inst.is_feasible(R, variant)
        r = exact.solve_cpsat(inst, variant, 'var', workers=2)
        assert abs(OBJECTIVES['var'](inst.row_sums(R)) - r['obj']) < 1e-6


@pytest.mark.parametrize('seed', range(6))
def test_heuristics_feasible_and_not_worse_than_tct(seed):
    inst = random_instance(8, 6, ['rect', 'cluster', 'scatter'][seed % 3], 0.35, levels=3, sigma=0.02, seed=seed)
    tct = OBJECTIVES['var'](inst.row_sums(inst.identity()))
    A = {'greedy': H.greedy_columns(inst), 'ra': H.ra(inst, restarts=5, seed=seed),
         'brave_ra': H.block_ra(inst, seed=seed), 'bra': H.block_ra(inst, rule='random', then_ra=False, seed=seed),
         'ils_A': H.ils(inst, 'A', budget_evals=3000, seed=seed)}
    B = {'bldm': H.bldm(inst), 'bwsa': H.bwsa(inst), 'ils_B': H.ils(inst, 'B', budget_evals=3000, seed=seed)}
    for name, R in A.items():
        assert inst.is_feasible(R, 'A'), name
    for name, R in B.items():
        assert inst.is_feasible(R, 'B'), name
    for name in ('ra', 'brave_ra', 'ils_A', 'ils_B'):
        R = {**A, **B}[name]
        assert OBJECTIVES['var'](inst.row_sums(R)) <= tct + 1e-9, name


def test_subset_sum_rows_feasible():
    inst = random_instance(4, 4, 'rect', 0.4, levels=3, seed=3)
    R = H.subset_sum_rows(inst)
    assert inst.is_feasible(R, 'B')


def test_ra_result_is_oppositely_ordered():
    """After RA terminates, every column is oppositely ordered to the rest (necessary
    optimality condition, Boudt et al. 2018)."""
    inst = random_instance(10, 7, 'cluster', 0.4, levels='continuous', seed=1)
    R = H.ra_single(inst, inst.identity())
    S = inst.row_sums(R)
    for j in range(inst.n):
        col = inst.flat[R[:, j]]; rest = S - col
        for a in range(inst.m):
            for b in range(inst.m):
                assert (col[a] - col[b]) * (rest[a] - rest[b]) <= 1e-9


def test_min_moves_relabel():
    inst = random_instance(5, 5, 'rect', 0.3, levels=2, seed=0)
    R = inst.identity()
    Rp = R[[4, 3, 2, 1, 0]]
    Rr, moves = H.min_moves_relabel(R, Rp)
    assert moves == 0 and np.array_equal(Rr, R)


@pytest.mark.parametrize('variant', ['A', 'B'])
def test_metaheuristics_feasible_and_budget(variant):
    from pvrearrange.metaheuristics import METAHEURISTICS
    inst = P26['central']
    for name, fn in METAHEURISTICS.items():
        r = fn(inst, variant, 'L1', budget=2000, seed=3)
        assert inst.is_feasible(r['R'], variant), name
        assert r['evals'] <= 2000, name
        assert abs(OBJECTIVES['L1'](inst.row_sums(r['R'])) - r['value']) < 1e-6, name


@pytest.mark.parametrize('variant', ['A', 'B'])
def test_cpsat_warm_start_not_worse_than_hint(variant):
    inst = random_instance(7, 6, 'cluster', 0.3, levels='continuous', seed=11)
    hint = H.ra(inst, restarts=3, seed=0)
    r = exact.solve_cpsat(inst, variant, 'var', time_limit=2, workers=2, hint=hint)
    assert r['obj'] <= OBJECTIVES['var'](inst.row_sums(hint)) + 1e-6
    assert inst.is_feasible(r['R'], variant)


# ---------------------------------------------------------------------------
# Proposed method: rearrangement with regrouping (variant B)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize('seed', range(4))
def test_rrg_feasible_budget_and_not_worse_than_bldm(seed):
    from pvrearrange.regroup import rrg
    inst = random_instance(9, 9, ['rect', 'cluster', 'scatter'][seed % 3], 0.3, levels='continuous', seed=seed)
    R = rrg(inst, 3000, seed=seed)
    assert inst.is_feasible(R, 'B')
    assert inst.row_sums(R).var() <= inst.row_sums(H.bldm(inst)).var() + 1e-9


def test_regroup_keeps_rows():
    from pvrearrange.regroup import regroup
    inst = random_instance(6, 5, 'scatter', 0.4, levels='continuous', seed=2)
    R = H.bldm(inst); rng = np.random.default_rng(0)
    for rule in ('sorted', 'random', 'mixed'):
        Rg = regroup(inst, R, rng, rule)
        assert all(sorted(Rg[i]) == sorted(R[i]) for i in range(inst.m))


def test_exchange_optimum_is_ra_fixed_point_under_regrouping():
    """Proposition 3: a local optimum of the pairwise exchange neighbourhood is not
    improved by RA under any regrouping."""
    from pvrearrange.regroup import regroup
    rng = np.random.default_rng(0)
    for s in range(5):
        inst = random_instance(7, 6, 'scatter', 0.4, levels='continuous', seed=s)
        R = H.local_search(inst, H.bldm(inst), 'B', 'var', rng=np.random.default_rng(s)); v = inst.row_sums(R).var()
        for _ in range(20):
            assert inst.row_sums(H.ra_single(inst, regroup(inst, R, rng, 'random'))).var() >= v - 1e-9


def test_rrg_exact_on_small_instances():
    from pvrearrange.regroup import rrg
    inst = random_instance(3, 3, 'scatter', 0.5, levels=3, seed=1)
    R = rrg(inst, 2000, seed=0)
    assert abs(inst.row_sums(R).var() - brute_force_B(inst, 'var')) < 1e-6
