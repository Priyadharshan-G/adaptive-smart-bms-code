#!/bin/bash
# Full BMS analysis for one pack (30 seeds, table plant, detector, capacity guard, figures, imputation).
# usage: tools/run_pack_analysis.sh LFP|NCR [pchip|poly]
#   LFP results in the paper use pchip; NCR uses poly (the paper's cubic, eq. 5).
# Inputs : packs/<PACK>/{Eco,Normal,Sport}_timeseries.csv + the pack's .slx at the repo root
# Outputs: packs/<PACK>/bms/results/   (about 20 min per pack)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACK=${1:?LFP or NCR}; OCV=${2:-poly}
BASE="$ROOT/packs/$PACK"; [ "$PACK" = LFP ] && [ "$OCV" = poly ] && BASE="$ROOT/packs/LFP_polynomial_OCV"
cd "$ROOT"
export ESS_BASE="$BASE" ESS_PACK="$PACK" ESS_OCV="$OCV" MPLBACKEND=Agg
python bms_pipeline_v2.py
python bms_robustness.py
python bms_revision2.py
python make_paper_figures.py
python make_extra_figures.py
python knn_imputation.py --can
echo "done: $BASE/bms/results"
