"""Problem instances for TCT PV array reconfiguration.

An instance is an m x n matrix G of module "current weights" (irradiance in W/m2,
optionally multiplied by a per-module ageing factor). Row i of a TCT array carries a
current proportional to the sum of the weights of the modules connected to it.

A solution is an m x n integer matrix R of flat module indices (p = k*n + j for the
module at physical row k, column j). Row i of the reconfigured array holds modules
R[i, :]. Variant 'A' (within-column) requires R[i, j] to come from column j;
variant 'B' (free balanced assignment) only requires every module to be used once.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
import numpy as np

K = 1000.0  # unshaded irradiance, W/m2


@dataclass
class Instance:
    name: str
    G: np.ndarray                      # m x n weights
    meta: dict = field(default_factory=dict)

    @property
    def m(self) -> int: return self.G.shape[0]
    @property
    def n(self) -> int: return self.G.shape[1]
    @property
    def flat(self) -> np.ndarray: return self.G.reshape(-1)

    def identity(self) -> np.ndarray:
        """The unreconfigured TCT arrangement."""
        return np.arange(self.m * self.n).reshape(self.m, self.n)

    def row_sums(self, R: np.ndarray) -> np.ndarray:
        return self.flat[R].sum(axis=1)

    def is_feasible(self, R: np.ndarray, variant: str) -> bool:
        R = np.asarray(R)
        if R.shape != (self.m, self.n): return False
        if sorted(R.reshape(-1).tolist()) != list(range(self.m * self.n)): return False
        if variant == 'A':
            return bool(np.all(R % self.n == np.arange(self.n)[None, :]))
        return True

    def to_json(self) -> dict:
        return {'name': self.name, 'G': self.G.tolist(), 'meta': self.meta}

    @staticmethod
    def from_json(d: dict) -> 'Instance':
        return Instance(d['name'], np.asarray(d['G'], float), d.get('meta', {}))


def save(instances, path):
    with open(path, 'w') as f:
        json.dump([i.to_json() for i in instances], f)


def load(path):
    with open(path) as f:
        return [Instance.from_json(d) for d in json.load(f)]


# ---------------------------------------------------------------------------
# Standard 6x6 patterns of Premkumar et al. (2026), Fig. 6
# ---------------------------------------------------------------------------
_k = K
PREMKUMAR_6x6 = {
    'none':          [[_k]*6]*6,
    'uneven_row':    [[400]*6] + [[_k]*6]*4 + [[600]*6],
    'uneven_column': [[400,_k,_k,_k,_k,400],[600,_k,_k,_k,_k,600],[400,_k,_k,_k,_k,600],
                      [600,_k,_k,_k,_k,400],[400,_k,_k,_k,_k,600],[600,_k,_k,_k,_k,400]],
    'diagonal':      [[300,_k,_k,_k,_k,500],[_k,400,_k,_k,500,_k],[_k,_k,500,_k,_k,_k],
                      [_k,_k,_k,300,_k,_k],[_k,400,_k,_k,400,_k],[300,_k,_k,_k,_k,500]],
    'short_narrow':  [[300,300,300,300,_k,_k],[400,400,400,400,_k,_k]] + [[_k]*6]*4,
    'short_wide':    [[400]*6,[600]*6,[700]*6] + [[_k]*6]*3,
    'long_narrow':   [[300,400,_k,_k,_k,_k],[300,500,_k,_k,_k,_k],[400,500,_k,_k,_k,_k],
                      [400,600,_k,_k,_k,_k],[500,600,_k,_k,_k,_k],[500,600,_k,_k,_k,_k]],
    'long_wide':     [[300,300,300,_k,_k,_k],[400,400,400,_k,_k,_k],[300,300,300,_k,_k,_k],
                      [400,400,400,_k,_k,_k],[500,500,500,_k,_k,_k],[500,500,500,_k,_k,_k]],
    'central':       [[_k]*6,[_k,400,300,300,500,_k],[_k,400,400,400,500,_k],
                      [_k,500,300,300,500,_k],[_k,400,400,400,500,_k],[_k]*6],
}


def premkumar_instances():
    return [Instance(f'P26_{k}', np.asarray(v, float), {'source': 'Premkumar2026', 'pattern': k})
            for k, v in PREMKUMAR_6x6.items()]


def scaled_standard(pattern: str, m: int, n: int) -> Instance:
    """Scale a 6x6 standard pattern to m x n by nearest-neighbour resampling."""
    P = np.asarray(PREMKUMAR_6x6[pattern], float)
    ri = np.minimum((np.arange(m) * 6) // m, 5)
    ci = np.minimum((np.arange(n) * 6) // n, 5)
    return Instance(f'std_{pattern}_{m}x{n}', P[np.ix_(ri, ci)].copy(),
                    {'source': 'scaled_standard', 'pattern': pattern, 'm': m, 'n': n})


# ---------------------------------------------------------------------------
# Random generators
# ---------------------------------------------------------------------------
def _levels(L, rng, lo=200.0, hi=800.0, spacing='equal'):
    """Shaded irradiance levels (unshaded level K is added separately)."""
    if L == 'continuous':
        return None
    if spacing == 'equal':
        return np.linspace(lo, hi, L)
    v = np.sort(rng.uniform(lo, hi, L))
    return np.round(v / 10) * 10


def random_instance(m, n, kind='rect', shaded_frac=0.25, levels=2, spacing='equal',
                    sigma=0.0, seed=0, n_shapes=None) -> Instance:
    """Random shading on an m x n array.

    kind: 'rect' (axis-aligned rectangular shadows), 'cluster' (blob grown from seeds)
          or 'scatter' (independent cells, i.e. dappled light).
    levels: number of shaded irradiance levels (int) or 'continuous'.
    sigma: std of a multiplicative ageing factor N(1, sigma) applied to every module.
    """
    rng = np.random.default_rng(seed)
    target = int(round(shaded_frac * m * n))
    mask = np.zeros((m, n), bool)
    if kind == 'scatter':
        idx = rng.choice(m * n, target, replace=False)
        mask.reshape(-1)[idx] = True
    elif kind == 'rect':
        shapes = n_shapes or int(rng.integers(1, 4))
        while mask.sum() < target:
            h = int(rng.integers(1, max(2, m // 2) + 1)); w = int(rng.integers(1, max(2, n // 2) + 1))
            r0 = int(rng.integers(0, m - h + 1)); c0 = int(rng.integers(0, n - w + 1))
            mask[r0:r0 + h, c0:c0 + w] = True
            shapes -= 1
            if shapes <= 0 and mask.sum() >= 0.5 * target: break
    elif kind == 'cluster':
        seeds = n_shapes or int(rng.integers(1, 4))
        frontier = [(int(rng.integers(m)), int(rng.integers(n))) for _ in range(seeds)]
        for r, c in frontier: mask[r, c] = True
        while mask.sum() < target:
            r, c = frontier[int(rng.integers(len(frontier)))]
            dr, dc = [(0, 1), (1, 0), (0, -1), (-1, 0)][int(rng.integers(4))]
            rr, cc = r + dr, c + dc
            if 0 <= rr < m and 0 <= cc < n and not mask[rr, cc]:
                mask[rr, cc] = True; frontier.append((rr, cc))
    else:
        raise ValueError(kind)

    G = np.full((m, n), K)
    lv = _levels(levels, rng, spacing=spacing)
    k = int(mask.sum())
    if lv is None:
        G[mask] = np.round(rng.uniform(200, 800, k))
    else:
        G[mask] = lv[rng.integers(0, len(lv), k)]   # each shaded cell gets a random level
    if sigma > 0:
        G = G * np.clip(rng.normal(1.0, sigma, (m, n)), 0.5, 1.5)
        G = np.round(G)
    meta = dict(kind=kind, shaded_frac=shaded_frac, levels=levels, spacing=spacing,
                sigma=sigma, seed=seed, m=m, n=n, shaded=int(k))
    return Instance(f'{kind}_{m}x{n}_f{shaded_frac}_L{levels}_{spacing}_s{sigma}_r{seed}', G, meta)


def features(inst: Instance) -> dict:
    """Instance features for the analysis of RQ4."""
    G = inst.G
    vals = np.unique(G)
    colvar = G.var(axis=0)
    return dict(m=inst.m, n=inst.n, n_values=int(len(vals)),
                shaded_frac=float((G < K).mean()),
                cv=float(G.std() / G.mean()),
                col_heterogeneity=float(colvar.std() / (colvar.mean() + 1e-12)),
                tct_range=float(np.ptp(G.sum(axis=1))))
