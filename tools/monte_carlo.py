"""Monte Carlo for the ground-truth test (G2): re-simulate the data from the TRUE equation with new random
seeds, re-run the developer's driver selection on each dataset, and record what it picks.

    python tools/monte_carlo.py --reps 50          # about 15 seconds per dataset

Uses the developer's own simulator (tools/simulate_data.py) unchanged except for the seed and output folder,
so every dataset comes from the same data-generating process as the one under review.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--dev", default=os.path.join(HERE, "..", "macroforecast"))
p.add_argument("--reps", type=int, default=50)
p.add_argument("--first-seed", type=int, default=1001)
p.add_argument("--out", default=os.path.join(HERE, "evidence", "monte_carlo_terms.csv"))
a = p.parse_args()

from macroforecast import Forecaster  # noqa: E402

src = open(os.path.join(a.dev, "tools", "simulate_data.py"), encoding="utf-8").read()
src = src.replace("SEED = 42", "import sys\nSEED = int(sys.argv[1])", 1)
src = "\n".join("OUT = sys.argv[2]" if line.startswith("OUT = ") else line for line in src.splitlines())
work = tempfile.mkdtemp(prefix="mc_")
sim = os.path.join(work, "sim.py")
open(sim, "w", encoding="utf-8").write(src)

rows = []
for seed in range(a.first_seed, a.first_seed + a.reps):
    d, out = os.path.join(work, f"d{seed}"), os.path.join(work, f"r{seed}")
    subprocess.run([sys.executable, sim, str(seed), d], check=True, capture_output=True)
    fc = Forecaster(target="loan_balance", path=out)
    fc.fit(data=os.path.join(d, "bank_data.csv"), metadata=os.path.join(d, "macro_metadata.csv"), literature=None,
           events=os.path.join(d, "bank_events.csv"), macro={"Baseline": os.path.join(d, "scenario_baseline.csv")},
           presets="fast", oos=False, trace=False)
    c = fc.coefficients()
    for m in ("Screen first", "Elastic Net first"):
        for _, r in c[c.model == m].iterrows():
            rows.append({"seed": seed, "model": m, "term": r.term, "coef": r.coef, "p_value": r.p_value})
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"dataset {seed}: done", flush=True)
shutil.rmtree(work, ignore_errors=True)
