#!/bin/bash
set -x
cd "$(dirname "$0")/.."
python scripts/server_data_export.py --out exports/figures --stage all
python scripts/export_nowcast_bootstrap.py --dir exports/figures
python scripts/make_bootstrap_inputs.py --dir exports/figures
Rscript R/make_figures.R exports/figures results/figures
Rscript R/paired_bootstrap.R --data-dir=exports/figures --out-dir=results/analysis --source-dir=results/source_data
echo FIGURES_DONE
