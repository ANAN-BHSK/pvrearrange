"""Surrogate objectives on the vector of row sums S (all Schur-convex, to be minimised)."""
import numpy as np


def v_var(S):   return float(np.var(S))                          # Shams El-Dein IMI / m^2
def v_l1(S):    return float(np.abs(S - np.mean(S)).sum())       # Premkumar et al. (2026)
def v_range(S): return float(np.max(S) - np.min(S))              # equalisation index, Velasco / Riva Sanseverino
def v_bot(S):   return float(-np.min(S))                         # limiting (bottleneck) row


OBJECTIVES = {'var': v_var, 'L1': v_l1, 'range': v_range, 'bot': v_bot}


def evaluate(inst, R, objective='var'):
    return OBJECTIVES[objective](inst.row_sums(R))


def all_objectives(inst, R):
    S = inst.row_sums(R)
    return {k: f(S) for k, f in OBJECTIVES.items()}
