"""POST-HOC diagnostics, run after the single test pass (never used for selection):
  1. does the validation ranking of the 17 designs predict their test ranking? (Spearman across designs)
  2. why did the frozen pick D05 fail? its residual-momentum sleeve run alone (diagnostic trial).
Output: output/diagnostics.md
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as R  # noqa: E402
import sv2  # noqa: E402


def main() -> None:
    logging.getLogger("setalpha").setLevel(logging.WARNING)
    val = pd.read_csv(sv2.OUT / "validation_all.csv", index_col=0)
    test = pd.read_csv(sv2.OUT / "test_all.csv", index_col=0).loc[val.index]
    rows = {}
    for c in [
        "Sharpe",
        "IR vs TDEX",
        "CAPM alpha t",
        "TH alpha t",
        "C4 alpha t",
        "CAGR %",
        "turnover 2-way x/yr",
        "TH b_SMB",
    ]:
        rho, p = spearmanr(val[c].astype(float), test[c].astype(float))
        rows[c] = {"Spearman rho (valid vs test, 17 designs)": rho, "p-value": p}
    rk = pd.DataFrame(rows).T
    inp = sv2.load_inputs()
    mom = sv2.Design(
        "diag: residual-momentum sleeve alone (4-weekly, buffer 30)",
        (sv2.Sleeve(score="resid_mom_120", buffer=30, every=4),),
        "diag",
        pickable=False,
    )
    e2 = next(d for d in sv2.grid(False) if d.name == R.E2_NAME)
    out = {}
    for split in ("valid", "test"):
        t, _, _, _ = R.run_designs([mom, e2], inp, split)
        for k in t.index:
            out[(split, k)] = t.loc[k]
    d = pd.DataFrame(out).T
    cols = [
        "CAGR %",
        "Sharpe",
        "turnover 2-way x/yr",
        "CAPM alpha %",
        "TH alpha %",
        "TH alpha t",
        "TH b_SMB",
        "TH t_SMB",
        "TH b_WML",
        "TH t_WML",
        "avg liq rank (w)",
        "vs EW top-100 % p.a.",
    ]
    text = (
        "# Post-hoc diagnostics (after the single test pass; not used for any selection)\n\n"
        "## 1. Does validation rank predict test rank across the 17 strategy_v2 designs?\n\n"
        + sv2.to_md(rk)
        + "\n\n## 2. Frozen pick D05 decomposed: momentum sleeve alone vs E2\n\n"
        + sv2.to_md(d[cols].astype(float))
        + "\n"
    )
    (sv2.OUT / "diagnostics.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
