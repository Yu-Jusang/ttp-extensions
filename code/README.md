# Code and reproduction

Simulation code for the two working papers in this repository:

- **Study A** — *Multi-source distributional test-then-pool* (`paper_A/`)
- **Study B** — *Shrinking equivalence radii and vanishing borrowing* (`paper_B/`)

Both build on Yang, Liu & Evans (2026), arXiv:2603.11867.

## Files

| file | content |
|---|---|
| `ttp.py` | kernel, Gram-matrix bookkeeping, permutation rows, p-values, minP, risk weights, MMD equivalence-fusion bootstrap, partial bootstrap |
| `methods.py` | all compared procedures (`study_A`, `study_B`); procedures share data, permutations and bootstrap weights |
| `run_sims.py` | simulation suites (`repro`, `repro_bw`, `A_main`, `A_n`, `B_n`, `B_local`) |
| `run_all.sh` | exploratory runs, base seed 20261018 |
| `run_conf.sh` | independent confirmation runs, base seed 20261019 |
| `analyze.py` | every table (`paper_*/gen/*.tex`) and figure (`paper_*/gen/*.pdf`) from the raw results |
| `pb_scalar.py` | Gaussian-limit check of the partial bootstrap at degeneracy (Study B, Conjecture) |

## Requirements

Python ≥ 3.10 with NumPy and Matplotlib (tested with Python 3.12.2, NumPy 2.5.4, Matplotlib 3.11.2). LaTeX (TeX Live 2024, `pdflatex`) to rebuild the PDFs.

## Running

```bash
cd code
bash run_all.sh      # exploratory suites
bash run_conf.sh     # confirmation suites
python analyze.py    # tables and figures
python pb_scalar.py  # partial-bootstrap limit check
cd ../paper_A && pdflatex paper_A.tex && pdflatex paper_A.tex
cd ../paper_B && pdflatex paper_B.tex && pdflatex paper_B.tex
```

`run_sims.py <suite> <reps_null> <reps_alt> <seed>` runs one suite with 15 worker processes (edit `Pool(15)` for your machine). Each data set is regenerated from `SeedSequence([seed, setting, level, replicate])`, so results do not depend on the number of workers. On a laptop with an Intel i7-1360P, the exploratory suites took about 2 hours and the confirmation suites about 1 hour.

## Raw results

`results/<suite>.npz` stores, for every setting and procedure:

- `p|<setting>|<method>`: p-values (0/1 decisions for partial-bootstrap procedures)
- `t|...`: timings in seconds
- `i|...`: diagnostics (selected weights, acceptance indicators, displacement checks)

`results/<suite>_meta.json` records the seed, replicate counts, elapsed time and software versions. `results/summary.json` collects summary numbers used in the text.
