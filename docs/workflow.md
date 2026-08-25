# Weekly workflow

## 1. Start the module

Open the module's `README.md` (e.g. `homework/02-dataframe-analysis/README.md`).
It links the upstream question set and the submission form.

Work on a branch so `main` always holds submitted work:

```bash
git checkout -b hw02
```

## 2. Work in the notebook

```bash
make lab
```

Open `homework/<module>/notebook.ipynb` and select the
**Python (stock-markets-zoomcamp)** kernel. Paste each question into its
markdown cell as you go — future-you will not remember what "Q3" was.

Reach for the shared helpers rather than rewriting them:

```python
from smaz import backtest, data, features, utils
```

## 3. Promote code that you'll reuse

If a function proves useful beyond this week, move it into `src/smaz/` and add a
test in `tests/`. That is what makes module 4 quick — and it feeds directly into
the project's "modular code" bonus point.

## 4. Record the answers

Fill in `answers.md`. Keep the working notes: they are what let you reproduce a
number when a peer reviewer or a leaderboard disagrees with you.

## 5. Verify before committing

```bash
make lint
make test
uv run jupyter nbconvert --to notebook --execute --inplace homework/<module>/notebook.ipynb
```

The last command proves the notebook runs top-to-bottom in a fresh kernel — the
single most common reason a submission can't be reproduced.

## 6. Commit and submit

```bash
git add homework/<module>
git commit -m "hw02: dataframe analysis"
git push -u origin hw02
```

The pre-commit hooks strip notebook outputs, run Ruff, and block large files.

Then submit on the course portal and tick the row in the root `README.md`
progress table.

## Capstone project

Same loop, but the deliverable is graded against a rubric. Work through
`project/notebooks/` in order (`01_eda` → `05_report`), promoting working code
into `project/src/algotrade/` as you go, then keep
[`project/README.md`](../project/README.md)'s scoring checklist honest — it is
the actual grading sheet, and each unticked line is a point left behind.

Before submitting, verify a cold clone reproduces:

```bash
uv run project/scripts/run_pipeline.py --mode file
```
