#!/bin/bash
# Exploratory suites (seed 20261018). Confirmation suites are run separately with seed 20261019.
set -e
cd "$(dirname "$0")"
python run_sims.py repro 1000 1000 20261018
python run_sims.py A_main 2000 1000 20261018
python run_sims.py A_n 2000 1000 20261018
python run_sims.py B_local 2000 1000 20261018
python run_sims.py B_n 2000 1000 20261018
