"""Build the developer's run that the validation reviews.

Clone the developer's repo next to this one first:
    git clone https://github.com/wuxiaokai-hk/macroforecast ../macroforecast
    pip install -e ../macroforecast
Then:
    python examples/make_run.py            # about 2 minutes, writes runs/loan_balance
"""
import argparse
import os

from macroforecast import Forecaster

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

p = argparse.ArgumentParser()
p.add_argument("--dev", default=os.path.join(HERE, "..", "macroforecast"), help="developer repo folder")
p.add_argument("--out", default=os.path.join(HERE, "runs", "loan_balance"))
p.add_argument("--presets", default="fast")
a = p.parse_args()
d = os.path.join(os.path.abspath(a.dev), "data")

fc = Forecaster(target="loan_balance", path=a.out)
fc.fit(data=os.path.join(d, "bank_data.csv"), metadata=os.path.join(d, "macro_metadata.csv"),
       literature=os.path.join(os.path.abspath(a.dev), "config", "literature_workbook.xlsx"),
       events=os.path.join(d, "bank_events.csv"), presets=a.presets, trace=False)
print(fc.leaderboard().head(8).to_string())
print("Recommended model:", fc.best_model)
