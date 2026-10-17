#!/bin/bash
# Confirmation suites with an independent base seed (20261019), fixed before inspecting them.
set -e
cd "$(dirname "$0")"
python run_sims.py repro_bw 1000 1000 20261019
python run_sims.py A_main_conf 2000 1000 20261019
python run_sims.py B_local_conf 2000 1000 20261019
python run_sims.py A_n_conf 1000 500 20261019
python run_sims.py B_n_conf 1000 500 20261019
