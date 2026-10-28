"""Electrical model of a TCT array: De Soto single-diode modules (pvlib), modules in a
row in parallel, rows in series, bypass diode clamping each row at VBYP.

Default module: AQT156PA-195W as used by Premkumar et al. (2026); 48 cells in series
reproduces their published TCT powers within 0.5 %.
"""
from __future__ import annotations
import numpy as np
from pvlib import ivtools, pvsystem

MODULE = dict(voc=33.25, isc=8.0, vmp=26.15, imp=7.48, cells=48,
              alpha_rel=0.0005, beta_rel=-0.0033, vbyp=-0.5, temp=25.0, noct=None)
# temp: uniform cell temperature (deg C). noct: if set (e.g. 45), cell temperature follows
# the NOCT model T = temp + (noct - 20) / 800 * G, so shaded modules run cooler.


class TCTModel:
    def __init__(self, module=None, n_grid=6000, v_points=1500):
        self.mod = dict(MODULE, **(module or {}))
        md = self.mod
        self.alpha = md['alpha_rel'] * md['isc']
        p, _ = ivtools.sdm.fit_desoto(md['vmp'], md['imp'], md['voc'], md['isc'],
                                      self.alpha, md['beta_rel'] * md['voc'], md['cells'])
        self.p = p
        self.V = np.linspace(md['vbyp'], md['voc'] * 1.05, v_points)
        self.n_grid = n_grid
        self._cache = {}

    def module_current(self, G):
        G = float(G)
        if G not in self._cache:
            md, p = self.mod, self.p
            T = md['temp'] if not md['noct'] else md['temp'] + (md['noct'] - 20) / 800 * G
            IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                G, T, self.alpha, p['a_ref'], p['I_L_ref'], p['I_o_ref'], p['R_sh_ref'], p['R_s'])
            self._cache[G] = np.clip(pvsystem.i_from_v(self.V, IL, I0, Rs, Rsh, nNsVth), 0, None)
        return self._cache[G]

    def pv_curve(self, rows):
        """rows: list of lists of module weights (W/m2). Returns (I, V, P) arrays."""
        vbyp = self.mod['vbyp']
        Irows = [sum(self.module_current(g) for g in row) for row in rows]
        Imax = max(I[0] for I in Irows)
        Igrid = np.linspace(0, Imax, self.n_grid)
        Varr = np.zeros_like(Igrid)
        for Irow in Irows:
            Vrow = np.interp(Igrid, Irow[::-1], self.V[::-1], left=self.V[-1], right=vbyp)
            Varr += np.maximum(Vrow, vbyp)
        return Igrid, Varr, Igrid * Varr

    def gmpp(self, rows):
        I, V, P = self.pv_curve(rows)
        k = int(np.argmax(P))
        o = np.argsort(V); pv = P[o]
        peaks = int(np.sum((pv[1:-1] > pv[:-2]) & (pv[1:-1] >= pv[2:]) & (pv[1:-1] > 0.02 * pv.max())))
        return dict(P=float(P[k]), V=float(V[k]), I=float(I[k]), peaks=peaks)

    def power(self, inst, R):
        return self.gmpp([inst.flat[r].tolist() for r in R])['P']


def power_local_search(inst, model, variant='A', starts=10, seed=0, max_evals=None):
    """H7: multi-start pairwise-swap local search on array power (Olanipekun et al. 2017),
    first improvement. Black-box objective, therefore much slower than surrogate methods."""
    from .heuristics import random_solution
    rng = np.random.default_rng(seed); m, n = inst.m, inst.n
    evals = 0; best = (-np.inf, None)
    for s in range(starts):
        R = inst.identity().copy() if s == 0 else random_solution(inst, variant, rng)
        cur = model.power(inst, R); evals += 1
        improved = True
        while improved:
            improved = False
            if variant == 'A':
                moves = [((a, j), (b, j)) for j in rng.permutation(n) for a in range(m) for b in range(a + 1, m)]
            else:
                cells = [(i, j) for i in range(m) for j in range(n)]
                moves = [(cells[x], cells[y]) for x in rng.permutation(len(cells)) for y in range(len(cells))
                         if cells[y][0] != cells[x][0]]
            for (a, ja), (b, jb) in moves:
                if max_evals and evals >= max_evals: break
                if inst.flat[R[a, ja]] == inst.flat[R[b, jb]]: continue
                R[a, ja], R[b, jb] = R[b, jb], R[a, ja]
                p = model.power(inst, R); evals += 1
                if p > cur + 1e-6: cur = p; improved = True; break
                R[a, ja], R[b, jb] = R[b, jb], R[a, ja]
        if cur > best[0]: best = (cur, R.copy())
        if max_evals and evals >= max_evals: break
    return best[1], best[0], evals
