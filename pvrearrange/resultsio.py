"""Reading the raw run results.

The experiment scripts write one CSV file per instance to results/full/ and
results/full_rrg/ (so that an interrupted run can be resumed). The published
repository stores each of these folders as one compressed file,
results/full_runs.csv.gz and results/full_rrg_runs.csv.gz. read_runs() reads the
per-instance files if they exist and the compressed file otherwise.
"""
import glob, os
import pandas as pd

FOLDERS = ('full', 'full_rrg')


def read_folder(name, root='results'):
    files = sorted(glob.glob(os.path.join(root, name, '*.csv')))
    if files:
        return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    packed = os.path.join(root, f'{name}_runs.csv.gz')
    if os.path.exists(packed):
        return pd.read_csv(packed)
    return pd.DataFrame()


def read_runs(folders=FOLDERS, root='results'):
    parts = [d for d in (read_folder(f, root) for f in folders) if len(d)]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def pack(root='results'):
    """Write results/<folder>_runs.csv.gz from the per-instance files."""
    for f in FOLDERS:
        files = sorted(glob.glob(os.path.join(root, f, '*.csv')))
        if files:
            pd.concat([pd.read_csv(x) for x in files], ignore_index=True).to_csv(
                os.path.join(root, f'{f}_runs.csv.gz'), index=False)


if __name__ == '__main__':
    pack()
