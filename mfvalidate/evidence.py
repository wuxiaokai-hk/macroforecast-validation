"""Load the developer's run and rebuild the model's data independently from the raw inputs.

The validator re-reads the raw files and re-implements the transforms itself, rather than reusing the
developer's prepared data. Where the two disagree, that is a finding (see checks/data.py, D1).
"""
import json
import os
import re

import numpy as np
import pandas as pd
import yaml

FEATURE = re.compile(r"^(?P<code>[^|]+)\|(?P<tr>[a-z0-9]+)\|L(?P<lag>\d+)$")
DUMMY = re.compile(r"^D_(\d{4}Q[1-4])$")


# ----------------------------------------------------------------------------------------- config
def load_config(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(path)

    def fix(p):
        return p if (p is None or os.path.isabs(p)) else os.path.normpath(os.path.join(base, p))

    cfg["_base"] = base
    cfg["run_dir"] = fix(cfg["run_dir"])
    for k in ("data", "metadata", "events"):
        cfg["inputs"][k] = fix(cfg["inputs"].get(k))
    cfg["inputs"]["scenarios"] = {k: fix(v) for k, v in cfg["inputs"]["scenarios"].items()}
    gt = cfg.get("ground_truth") or {}
    if gt.get("monte_carlo_results"):
        gt["monte_carlo_results"] = fix(gt["monte_carlo_results"])
    return cfg


# ----------------------------------------------------------------------------------------- transforms
def transform(x, tr):
    """Independent implementation of the developer's stationary transforms (macro series in levels)."""
    if tr == "lvl":
        return x
    if tr == "diff":
        return x.diff()
    if tr in ("diff2", "diff4"):
        return x.diff(int(tr[-1]))
    if tr == "rgap":
        return x - x.rolling(12, min_periods=12).mean()
    lx = np.log(x)
    if tr == "dlog":
        return lx.diff()
    if tr == "dlog2":
        return lx.diff(2)
    if tr == "yoy":
        return lx.diff(4)
    if tr == "acc":
        return lx.diff(4) - lx.diff(4).shift(4)
    if tr == "gap":
        return lx - lx.rolling(12, min_periods=12).mean()
    raise NotImplementedError(f"transform '{tr}' is not re-implemented by the validator")


def parse_feature(name):
    m = FEATURE.match(name)
    if not m:
        return None
    return m["code"], m["tr"], int(m["lag"])


def feature(macro, name):
    code, tr, lag = parse_feature(name)
    return transform(macro[code].astype(float), tr).shift(lag)


# ----------------------------------------------------------------------------------------- raw data
def read_quarterly(path):
    df = pd.read_csv(path)
    df.index = pd.PeriodIndex(df.pop(df.columns[0]).astype(str), freq="Q")
    return df


def organic_growth(level, events, target):
    """Quarterly log growth with inorganic events removed: ln((L_t - event_t) / L_{t-1})."""
    adj = pd.Series(0.0, index=level.index)
    if events is not None:
        ev = events[events["series"] == target] if "series" in events else events
        for _, r in ev.iterrows():
            q = pd.Period(str(r["date"]), freq="Q")
            if q in adj.index:
                adj[q] += float(r["amount"])
    return np.log((level - adj) / level.shift(1)), adj


class Evidence:
    """Everything the checks need, gathered once."""

    def __init__(self, cfg):
        from macroforecast import Forecaster  # the developer's package, used read-only

        self.cfg = cfg
        self.run_dir = cfg["run_dir"]
        self.fc = Forecaster.load(self.run_dir)
        with open(os.path.join(self.run_dir, "run_config.yaml"), encoding="utf-8") as f:
            self.run_config = yaml.safe_load(f)
        with open(os.path.join(self.run_dir, "macroforecast_result.json"), encoding="utf-8") as f:
            self.result = json.load(f)
        m = cfg.get("model_under_review", "auto")
        self.model = self.result["best"]["best_model"] if m in (None, "auto") else m

        inp = cfg["inputs"]
        self.target = inp["target"]
        raw = read_quarterly(inp["data"])
        self.level = raw[self.target].astype(float)
        self.events = pd.read_csv(inp["events"]) if inp.get("events") else None
        self.growth, self.event_adj = organic_growth(self.level, self.events, self.target)
        self.meta = pd.read_csv(inp["metadata"]).set_index("mnemonic")
        self.scenarios = {k: read_quarterly(v) for k, v in inp["scenarios"].items()}
        self.base_name = next(iter(self.scenarios))
        s = self.run_config["sample"]
        self.hist_end = pd.Period(s["hist_end"], freq="Q")
        self.est_start = pd.Period(s["estimation_start"], freq="Q")
        self.macro_hist = self.scenarios[self.base_name].loc[:self.hist_end]

        coefs = self.fc.coefficients(self.model)
        self.coefs = coefs.set_index("term")["coef"].astype(float)
        self.coef_table = coefs
        self.terms = list(self.coefs.index)
        self.drivers = [t for t in self.terms if parse_feature(t)]
        self.dummies = [t for t in self.terms if DUMMY.match(t)]
        self.has_ar = "y_lag1" in self.terms

    # ---------------------------------------------------------------- design matrix
    def design(self, macro=None, growth=None, terms=None):
        """Regressors of the model under review, built from raw data by the validator."""
        macro = self.macro_hist if macro is None else macro
        growth = self.growth if growth is None else growth
        terms = self.terms if terms is None else terms
        X = pd.DataFrame(index=macro.index)
        for t in terms:
            if t == "const":
                X[t] = 1.0
            elif t == "y_lag1":
                X[t] = growth.reindex(macro.index).shift(1)
            elif DUMMY.match(t):
                X[t] = (X.index == pd.Period(DUMMY.match(t)[1], freq="Q")).astype(float)
            elif parse_feature(t):
                X[t] = feature(macro, t)
            else:
                raise NotImplementedError(f"term '{t}' is not re-implemented by the validator")
        return X

    def estimation_frame(self):
        X = self.design()
        y = self.growth.reindex(X.index)
        df = X.assign(_y=y).loc[self.est_start:self.hist_end].dropna()
        return df["_y"], df.drop(columns="_y")

    # ---------------------------------------------------------------- developer outputs
    def run_file(self, *parts):
        p = os.path.join(self.run_dir, *parts)
        return pd.read_csv(p) if os.path.exists(p) else None
