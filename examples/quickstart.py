"""Minimal example: one random 16x16 shaded array, solved with the fixed TCT connection,
block rearrangement (BRA, variant A), the proposed RRG (variant B) and CP-SAT (variant A, 10 s).

Run from the repository root:  python examples/quickstart.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from pvrearrange import random_instance, exact
from pvrearrange import heuristics as H
from pvrearrange.regroup import rrg
from pvrearrange.power import TCTModel

inst = random_instance(16, 16, kind='cluster', shaded_frac=0.25, levels=3, seed=1)
sd = lambda R: inst.row_sums(R).std()            # standard deviation of the row totals (W/m^2)
model = TCTModel()                                # reference electrical model (De Soto, pvlib)

sols = {
    'TCT (no reconfiguration)': inst.identity(),
    'BRA, variant A': H.block_ra(inst, rule='random', then_ra=False, iters=10_000, seed=0),
    'RRG, variant B': rrg(inst, budget_evals=10_000, seed=0, rule='random',
                          block_iters=max(3, round(1.096 * inst.n))),     # tuned parameters
}
cp = exact.solve_cpsat(inst, 'A', 'var', time_limit=10)   # single thread, no warm start
sols[f"CP-SAT, variant A ({cp['status']})"] = cp['R']

for name, R in sols.items():
    print(f'{name:32s} sd(S) = {sd(R):7.1f} W/m2   power = {model.power(inst, R):8.1f} W')
