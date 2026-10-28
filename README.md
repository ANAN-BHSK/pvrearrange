# pvrearrange

Code and data for the paper

> A. Banharnsakun, *Exploiting matrix rearrangement structure in photovoltaic array reconfiguration under partial shading*, submitted to Computers & Operations Research.

The repository contains the instance generator, every method compared in the paper (exact model, classical heuristics, the proposed rearrangement with regrouping (RRG), metaheuristics), the tuning procedure, the electrical model, the raw results of every run, and the scripts that produce all tables and figures.

## The problem in brief

An m × n total-cross-tied (TCT) photovoltaic array is reconfigured by deciding which modules are connected to each electrical row. The goal is to make the row totals of module irradiance as equal as possible.

- **Variant A:** modules stay in their physical column. This is the *matrix rearrangement problem*.
- **Variant B:** any module may be connected to any row, with n modules per row. This is *balanced multiway number partitioning*.

## Installation

Python 3.10 or newer.

```
pip install -r requirements.txt
python -m pytest -q tests          # 50 correctness tests, about 10 s
python examples/quickstart.py      # one 16×16 instance: TCT, BRA, RRG and CP-SAT
```

## Repository layout

| Path | Content |
|---|---|
| `pvrearrange/instance.py` | `Instance` class and solution representation. Also: the nine 6×6 shading patterns of Premkumar et al. (2026), scaled standard patterns, the random instance generator (rect / cluster / scatter shadows, number of levels, aging spread σ_a), and instance features. |
| `pvrearrange/objectives.py` | Surrogate objectives on the row totals: `var`, `L1`, `range`, `bot`. |
| `pvrearrange/exact.py` | **Exact methods.**<br>E1: CP-SAT model for variants A and B, with count variables and row-order symmetry breaking.<br>E4: the optimal algorithm for two-valued instances (Proposition 2). |
| `pvrearrange/heuristics.py` | **Classical heuristics.**<br>H1: greedy column ordering.<br>H2: Rearrangement Algorithm (RA).<br>H3: block rearrangement (BRA) and BRAVE+RA.<br>H4: balanced largest differencing (BLDM).<br>H5: BWSA.<br>H6: swap local search and ILS. |
| `pvrearrange/regroup.py` | **Proposed method:** rearrangement with regrouping (RRG, Algorithm 1). |
| `pvrearrange/metaheuristics.py` | Random-key methods with a shared evaluation budget:<br>M0 random search, M1 RKGA, M2 PSO, M3 DE, M4 GOA, M5 HHO. |
| `pvrearrange/power.py` | TCT array power: De Soto single-diode model with bypass diodes (pvlib), and local search on the power. |
| `pvrearrange/resultsio.py` | Reading the raw run results (per-instance files or the compressed files in `results/`). |
| `pvrearrange/racing.py` | Iterated racing (the irace procedure, re-implemented) used for parameter tuning. |
| `scripts/` | Experiment, analysis and figure scripts (see below). |
| `tests/test_methods.py` | Correctness tests. They check CP-SAT against brute force for all objectives and both variants, the feasibility of every method, and the RA optimality condition. |
| `examples/quickstart.py` | Minimal usage example. |
| `run_all.py` | One-command runner for the whole study (see below). |
| `results/` | Raw results and analysis output (see below). |

### Solution representation

`R[i, j]` is the flat index `k*n + j'` of the module placed at position j of electrical row i, so `inst.row_sums(R)` gives the row totals. In variant A, `j' == j` (each module stays in its column). The unreconfigured TCT arrangement is `inst.identity()`.

## Reproducing the study

Everything below runs from the repository root.

### Recompute all tables and figures from the stored raw results (about 2 minutes)

```
python scripts/analyse_replication.py
python scripts/analyse_full.py
python scripts/report_data.py
python scripts/manuscript_data.py
python scripts/figures.py
python scripts/figures_method.py
python scripts/figure_rrg_array.py 6
python scripts/figure_rrg_array.py
```

### Rerun the experiments

The full study can be rerun with one command:

```
python run_all.py --mode full      # several hours, depending on the number of cores
python run_all.py --mode cpsat     # long CP-SAT runs (600 s, 8 threads, warm start)
```

`--mode full` runs these steps in order:
1. Sensitivity study / RQ3
2. Replication of Premkumar et al. (2026)
3. Tuning
4. Full test run
5. RRG and BLDM+ILS runs
6. RRG component analysis
7. Analysis
8. Figures

`--mode cpsat` re-solves every instance–variant pair not proven optimal in the 60 s runs. It is restarted with the same command if interrupted. Afterwards, rerun the analysis scripts above. `python run_all.py -h` lists all options.

Single steps can also be run directly:

| Script | Paper section | Output |
|---|---|---|
| `scripts/replicate_premkumar.py`, `scripts/analyse_replication.py` | 8.1 | `results/replication_*.csv` |
| `scripts/tune.py 400` | 7.3 | `results/tuned_params.json`, `results/tuning_history.json` |
| `scripts/run_full.py <procs>` | 7, 8.2–8.4 | `results/full/*.csv` (one file per instance) |
| `scripts/run_rrg.py <procs>` | 6.4, 8.3 | `results/full_rrg/*.csv` |
| `scripts/ablation_rrg.py <procs>` | 8.7 | `results/ablation_rrg.csv` |
| `scripts/sensitivity_rq3.py` | 8.5 | `results/sensitivity_power.csv`, `results/rq3_spearman.csv` |
| `scripts/analyse_full.py` | 8 | `results/analysis/` (statistics, best-known values, gaps) |
| `scripts/report_data.py`, `scripts/manuscript_data.py` | all | `results/analysis/report_data.json`, `results/analysis/manuscript_data.json` |

Stochastic methods use seeds 0–29, so the runs are repeatable. Run times differ between computers.

## Where each table and figure comes from

| Paper | Source |
|---|---|
| Fig. 1–4 | `scripts/figures_method.py` → `results/figures/figM1_overview.png`, `figM2_problem.png`, `figM3_rrg_example.png`, `figM4_rrg_flowchart.png` |
| Fig. 5 | `scripts/figures.py` → `results/figures/fig1_cpsat.png` |
| Fig. 6 | `scripts/figures.py` → `results/figures/fig2_gap_profile.png` |
| Fig. 7 | `scripts/figures.py` → `results/figures/fig3_anytime.png` |
| Fig. 8 | `scripts/figure_rrg_array.py 6` → `results/figures/figR_rrg_array_6x6.png` |
| Fig. 9 | `scripts/figure_rrg_array.py` → `results/figures/figR_rrg_array.png` |
| Table 2 | `results/tuned_params.json` |
| Table 3 | `results/analysis/manuscript_data.json` → `replication` |
| Table 4 | `results/analysis/report_data.json` → `cpsat` (from `rq1_cpsat_combined.csv`) |
| Tables 5 and 6 | `results/analysis/rq2_methods.csv` |
| Table 7 | `results/analysis/rq2_gap_by_levels.csv`, `rq2_gap_by_size.csv` |
| Table 8 | `results/analysis/report_data.json` → `sensitivity` (from `results/sensitivity_power.csv`) |
| Table 9 | `results/analysis/tuned_vs_default.csv` |
| Table 10 | `results/analysis/manuscript_data.json` → `ablation` (from `results/ablation_rrg.csv`) |
| Mixed-effects models | `results/analysis/rq4_mixedlm_A.csv`, `rq4_mixedlm_B.csv` |

## Results files

**Instance names** encode the generator call, e.g. `cluster_16x16_f0.25_L3_equal_s0.02_r3216`:
- `cluster`: shadow type
- `16x16`: m × n
- `f0.25`: shaded fraction
- `L3`: number of levels (`Lcontinuous` = continuous irradiance)
- `equal`: equally spaced levels
- `s0.02`: aging spread σ_a
- `r3216`: seed

`scripts/run_full.py` (`design()`) lists all 118 test instances. They fall into groups S1–S4; S4 holds the Premkumar patterns, named `std_<pattern>_<m>x<m>`.

**`results/full_runs.csv.gz`, `results/full_rrg_runs.csv.gz`** have one row per run (all methods except RRG and BLDM+ILS, and those two methods, respectively). The experiment scripts write the same rows as one file per instance to `results/full/` and `results/full_rrg/`, so that an interrupted run can be resumed; `pvrearrange/resultsio.py` reads the per-instance files if they exist and the compressed files otherwise (`python -m pvrearrange.resultsio` packs them). Columns:
- Instance: `instance`, `set`, `m`, `n`, `kind`, `levels`, `frac`, `sigma`
- Instance features: `n_values` (number of distinct weights), `cv`, `col_het`
- Run identity: `variant` (A or B), `method`, `params` (`tuned`, `default`, or `-` for parameter-free methods), `seed`
- Run outcome: `status`, `time` (seconds), `evals_to_best`
- Surrogate values of the final solution: `var`, `L1`, `range`, `bot`
- CP-SAT only: `bound_var`, the lower bound on the variance
- Power: `power` (array power in W, reference model) and `peaks` (number of local maxima of the P–V curve)
- `mean_row` (the mean row total μ)
- Best variance after 1,000, 2,500, 5,000 and 10,000 evaluations: `var@1000` … `var@10000`

**`results/cpsat_long_all.csv`** holds the long CP-SAT runs (600 s, 8 threads, warm start), one row per instance–variant pair.
- `best_var`: the variance of the final solution.
- `bound_var`: the solver's lower bound.
- `hint_var`, `hint_method`: the warm start.
- `previous_status`, `previous_obj_var`: the result of the 60 s run.

**`results/analysis/all_runs_with_gap.csv.gz`** holds all runs, with these columns added:
- `sd_best`: the best-known standard deviation σ*.
- `gap`: (σ_S − σ*)/μ × 100.
- `hit`: whether the run reached the best-known value.
- `proven`: whether σ* is proven optimal.
- `power_gain`: the power gain over the TCT array.

`results/analysis_before_long_cpsat/` keeps the same file as computed before the long CP-SAT runs. It is used only to report how much those runs changed the best-known values.

**`results/second_machine/`** repeats the tuning and the RQ3 / sensitivity analysis on a second computer (Section 8.5).

## Citation

If you use this code or data, please cite the paper (citation details will be added on publication) and this repository; see `CITATION.cff`.

## License

MIT License, see `LICENSE`.
