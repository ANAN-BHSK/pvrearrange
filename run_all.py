"""One-command runner: checks the environment, runs the correctness tests, runs the
experiment, and packs every result into one zip file.

Usage (from the pvrearrange folder):
    python run_all.py --mode full        # the whole study (sensitivity, replication, tuning, test runs, analysis, figures)
    python run_all.py --mode cpsat       # long CP-SAT runs on every case not proven optimal in 60 s (default mode)
    python run_all.py --time-limit 300   # shorter CP-SAT limit per case (default 600 s)

Options:
    --mode cpsat|full     what to run (default cpsat)
    --time-limit S        CP-SAT time limit per case in seconds (default 600)
    --procs P             parallel CP-SAT processes (default: 1 per 8 cores)
    --threads T           CP-SAT threads per process (default: cores / procs)
    --all-instances       cpsat mode: also re-run cases already proven optimal
    --limit N             only the first N cases (for a quick trial run)

The run can be stopped (Ctrl+C) and started again with the same command; finished cases
are kept. When it ends, the path of the zip file is printed.
"""
import argparse, csv, glob, importlib, json, os, platform, subprocess, sys, time, zipfile
from datetime import datetime

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
NEEDED = {'numpy': 'numpy', 'scipy': 'scipy', 'pandas': 'pandas', 'ortools': 'ortools', 'pvlib': 'pvlib',
          'pytest': 'pytest', 'statsmodels': 'statsmodels', 'matplotlib': 'matplotlib'}
LOG = os.path.join(ROOT, 'results', 'run_all.log')


def log(msg):
    line = f'[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}'
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, 'a', encoding='utf-8') as fh: fh.write(line + '\n')


def ensure_packages():
    missing = [pip for mod, pip in NEEDED.items() if importlib.util.find_spec(mod) is None]
    if missing:
        log(f'installing missing packages: {" ".join(missing)}')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', *missing])
    importlib.invalidate_caches()


def machine_info(args):
    import numpy, scipy, pandas, ortools, pvlib
    info = dict(started=datetime.now().isoformat(timespec='seconds'), host=platform.node(),
                os=platform.platform(), python=platform.python_version(), processor=platform.processor(),
                cpu_count=os.cpu_count(), args=vars(args),
                versions=dict(numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__,
                              ortools=ortools.__version__, pvlib=pvlib.__version__))
    try:
        import ctypes
        if platform.system() == 'Windows':
            class MS(ctypes.Structure):
                _fields_ = [('l', ctypes.c_ulong), ('m', ctypes.c_ulong), ('t', ctypes.c_ulonglong), ('a', ctypes.c_ulonglong),
                            ('tp', ctypes.c_ulonglong), ('ap', ctypes.c_ulonglong), ('tv', ctypes.c_ulonglong),
                            ('av', ctypes.c_ulonglong), ('ae', ctypes.c_ulonglong)]
            s = MS(); s.l = ctypes.sizeof(MS); ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s))
            info['ram_gb'] = round(s.t / 2 ** 30, 1)
        else:
            info['ram_gb'] = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2 ** 30, 1)
    except Exception:
        pass
    return info


def run_tests():
    log('running correctness tests ...')
    r = subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests'], capture_output=True, text=True)
    tail = (r.stdout or r.stderr).strip().splitlines()[-1:]
    log(f'tests: {tail[0] if tail else "no output"}')
    with open(os.path.join(ROOT, 'results', 'tests_output.txt'), 'w', encoding='utf-8') as fh: fh.write(r.stdout + r.stderr)
    if r.returncode != 0:
        log('TESTS FAILED - stopping. See results/tests_output.txt.')
        pack(['results/tests_output.txt', 'results/run_all.log', 'results/machine_info.json'], 'FAILED_TESTS')
        sys.exit(1)


# ---------------------------------------------------------------------------
# Mode 1: long CP-SAT re-run
# ---------------------------------------------------------------------------
CP_OUT = os.path.join('results', 'cpsat_long')
CP_FIELDS = ['instance', 'variant', 'status', 'best_var', 'obj_var', 'bound_var', 'time_s', 'time_limit_s', 'threads',
             'hint_var', 'hint_method', 'previous_status', 'previous_obj_var']


def _cases(args):
    import pandas as pd
    import run_full as rf
    prev = {}
    from pvrearrange.resultsio import read_folder
    d = read_folder('full')
    if len(d):
        d = d[d.method == 'CP-SAT']
        prev = {(r.instance, r.variant): (r.status, r['var']) for _, r in d.iterrows()}
    cases = []
    for spec in rf.design():
        st, inst = rf.make(spec)
        for v in 'AB':
            ps, pv = prev.get((inst.name, v), ('', ''))
            if ps == 'OPTIMAL' and not args.all_instances: continue
            cases.append((spec, inst.name, v, ps, pv, inst.m * inst.n))
    cases.sort(key=lambda c: c[5])                     # small first: early results come quickly
    return cases[:args.limit] if args.limit else cases


def _solve_case(job):
    spec, name, v, ps, pv, _, tl, th = job
    import numpy as np
    import run_full as rf
    from pvrearrange import exact, heuristics as H
    path = os.path.join(CP_OUT, f'{name}__{v}.csv')
    if os.path.exists(path): return name, v, 'skipped', 0.0
    st, inst = rf.make(spec)
    # warm start from a fast heuristic (recorded, so the run is transparent)
    cand = []                                          # best of several fast heuristics, 3 seeds each
    for sd in range(3):
        cand.append((H.block_ra(inst, rule='random', then_ra=False, iters=10_000, seed=sd), 'BRA'))
        cand.append((rf.ra_traced(inst, sd)[0], 'RA'))
        cand.append((H.ils(inst, 'A', 'var', 10_000, kick=4, seed=sd), 'ILS-A'))
    if v == 'B':                                       # every variant-A solution is feasible for B
        c1 = H.bldm(inst); cand.append((c1, 'BLDM'))
        cand.append((H.ils(inst, 'B', 'var', 10_000, kick=4, seed=0, R0=c1), 'BLDM+ILS'))
    hint, hm = min(cand, key=lambda t: inst.row_sums(t[0]).var())
    hv = float(inst.row_sums(hint).var())
    r = exact.solve_cpsat(inst, v, 'var', time_limit=tl, workers=th, hint=hint)
    best = hv if r['obj'] is None else min(hv, r['obj'])
    row = dict(instance=name, variant=v, status=r['status'], best_var=round(best, 6), obj_var=r['obj'], bound_var=r['bound'],
               time_s=round(r['time'], 2), time_limit_s=tl, threads=th, hint_var=round(hv, 6), hint_method=hm,
               previous_status=ps, previous_obj_var=pv)
    tmp = path + '.tmp'
    with open(tmp, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CP_FIELDS); w.writeheader(); w.writerow(row)
    os.replace(tmp, path)
    return name, v, r['status'], r['time']


def mode_cpsat(args):
    from multiprocessing import Pool
    os.makedirs(CP_OUT, exist_ok=True)
    cores = os.cpu_count() or 1
    procs = args.procs or max(1, cores // 8)
    threads = args.threads or max(1, cores // procs)
    cases = _cases(args)
    todo = [c for c in cases if not os.path.exists(os.path.join(CP_OUT, f'{c[1]}__{c[2]}.csv'))]
    worst_h = len(todo) * args.time_limit / procs / 3600
    log(f'CP-SAT re-run: {len(cases)} cases ({len(todo)} still to do), {procs} process(es) x {threads} threads, '
        f'limit {args.time_limit} s per case -> at most about {worst_h:.1f} h')
    jobs = [(*c, args.time_limit, threads) for c in todo]
    done = len(cases) - len(todo)
    with Pool(procs) as pool:
        for name, v, status, t in pool.imap_unordered(_solve_case, jobs):
            done += 1
            log(f'{done}/{len(cases)} {name} [{v}] {status} {t:.0f}s')
    rows = []
    for f in sorted(glob.glob(os.path.join(CP_OUT, '*.csv'))):
        with open(f) as fh: rows += list(csv.DictReader(fh))
    with open(os.path.join('results', 'cpsat_long_all.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=CP_FIELDS); w.writeheader(); w.writerows(rows)
    n_opt = sum(r['status'] == 'OPTIMAL' for r in rows)
    log(f'finished: {n_opt}/{len(rows)} cases proven optimal')
    return ['results/cpsat_long_all.csv', CP_OUT]


# ---------------------------------------------------------------------------
# Mode 2: whole study
# ---------------------------------------------------------------------------
def mode_full(args):
    procs = args.procs or max(1, (os.cpu_count() or 2))
    steps = [('sensitivity and RQ3', ['scripts/sensitivity_rq3.py']),
             ('replication of Premkumar et al.', ['scripts/replicate_premkumar.py']),
             ('replication analysis', ['scripts/analyse_replication.py']),
             ('parameter tuning', ['scripts/tune.py', '400']),
             ('full test run', ['scripts/run_full.py', str(procs)]),
             ('proposed method RRG', ['scripts/run_rrg.py', str(procs)]),
             ('RRG component analysis', ['scripts/ablation_rrg.py', str(procs)]),
             ('statistical analysis', ['scripts/analyse_full.py']),
             ('report numbers', ['scripts/report_data.py']),
             ('manuscript numbers', ['scripts/manuscript_data.py']),
             ('figures', ['scripts/figures.py']),
             ('method figures', ['scripts/figures_method.py']),
             ('reconfiguration example 6x6', ['scripts/figure_rrg_array.py', '6']),
             ('reconfiguration example 9x9', ['scripts/figure_rrg_array.py'])]
    for title, cmd in steps:
        log(f'step: {title} ...'); t0 = time.time()
        with open(os.path.join('results', f'log_{os.path.basename(cmd[0])}.txt'), 'w', encoding='utf-8') as fh:
            r = subprocess.run([sys.executable, *cmd], stdout=fh, stderr=subprocess.STDOUT)
        log(f'step: {title} finished in {time.time() - t0:.0f} s (exit code {r.returncode})')
        if r.returncode != 0:
            log(f'step {title} FAILED - see results/log_{os.path.basename(cmd[0])}.txt'); break
    return ['results']


# ---------------------------------------------------------------------------
def pack(paths, tag):
    name = f'pv_results_{tag}_{platform.node() or "pc"}_{datetime.now():%Y%m%d-%H%M}.zip'
    out = os.path.join(ROOT, name)
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in paths + ['results/run_all.log', 'results/machine_info.json', 'results/tests_output.txt']:
            if os.path.isdir(p):
                for dp, _, fs in os.walk(p):
                    for f in fs:
                        if not f.endswith('.tmp'): z.write(os.path.join(dp, f))
            elif os.path.exists(p):
                z.write(p)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--mode', choices=['cpsat', 'full'], default='cpsat')
    ap.add_argument('--time-limit', type=float, default=600.0)
    ap.add_argument('--procs', type=int, default=0)
    ap.add_argument('--threads', type=int, default=0)
    ap.add_argument('--all-instances', action='store_true')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    log(f'=== run_all.py mode={args.mode} ===')
    ensure_packages()
    os.makedirs('results', exist_ok=True)
    json.dump(machine_info(args), open('results/machine_info.json', 'w'), indent=1)
    run_tests()
    t0 = time.time()
    paths = mode_cpsat(args) if args.mode == 'cpsat' else mode_full(args)
    log(f'total time {(time.time() - t0) / 3600:.2f} h')
    out = pack(paths, args.mode)
    log('=' * 60)
    log(f'DONE. All results are packed in:\n    {out}')


if __name__ == '__main__':
    main()
