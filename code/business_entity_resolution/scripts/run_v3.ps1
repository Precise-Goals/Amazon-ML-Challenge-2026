# v3 end-to-end: honest full-universe training -> two test submissions -> report.
# Run from the repo root (the folder that contains datasource/):
#   powershell -ExecutionPolicy Bypass -File code/business_entity_resolution/scripts/run_v3.ps1
# Optional quick honest check first (one country, much faster):
#   python code/business_entity_resolution/src/train_universe.py --countries India
$ErrorActionPreference = "Continue"  # native stderr (LightGBM warnings) must not abort; exit codes are checked
$src = "code/business_entity_resolution/src"
New-Item -ItemType Directory -Force reports | Out-Null

python -m pip install -q -r code/business_entity_resolution/requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

# 1) Train on the FULL train universe (writes models/v3, appends reports/experiments.jsonl)
python "$src/train_universe.py" 2>&1 | Tee-Object -FilePath reports/v3_train_console.log
if ($LASTEXITCODE -ne 0) { throw "training failed - send reports/v3_train_console.log" }

# 2) Test universe + submission A (tuned thresholds, no extra caution for unseen countries)
python "$src/predict_universe.py" --tag base 2>&1 | Tee-Object -FilePath reports/v3_predict_console.log
if ($LASTEXITCODE -ne 0) { throw "prediction failed - send reports/v3_predict_console.log" }

# 3) Submission B: reuse the built test universe, +0.05 threshold for countries unseen in train (France)
python "$src/predict_universe.py" --skip-build --unseen-offset 0.05 --tag unseen05 2>&1 | Tee-Object -Append -FilePath reports/v3_predict_console.log

# 4) One small file to send back
python "$src/make_v3_report.py"
Write-Host "DONE. Send reports/v3_report_for_claude.md. Files: output/matching_results_base.tsv, output/matching_results_unseen05.tsv"
