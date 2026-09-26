# Time-boxed v3 run: train universe and test universe built IN PARALLEL (if RAM allows),
# then training, two submissions and the report. Every country is checkpointed, so a crash or
# a rerun resumes where it stopped. Run from the repo root (folder containing datasource/):
#   powershell -ExecutionPolicy Bypass -File code/business_entity_resolution/scripts/run_v3_6h.ps1
$ErrorActionPreference = "Continue"
$src = "code/business_entity_resolution/src"
New-Item -ItemType Directory -Force reports | Out-Null
python -m pip install -q -r code/business_entity_resolution/requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }

$ramGB = [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
$parallel = $ramGB -ge 24
Write-Host "RAM ${ramGB} GB -> parallel test-universe build: $parallel"

$testJob = $null
if ($parallel) {
    $testJob = Start-Process python -ArgumentList "$src/predict_universe.py --build-only" `
        -RedirectStandardOutput reports/v3_test_build.log -RedirectStandardError reports/v3_test_build.err `
        -NoNewWindow -PassThru
}

# Train universe: smaller country first so a fallback model exists as early as possible.
python "$src/train_universe.py" --countries India US 2>&1 | Tee-Object -FilePath reports/v3_train_console.log
if ($LASTEXITCODE -ne 0) { throw "training failed - send reports/v3_train_console.log" }

if ($parallel) {
    Write-Host "waiting for the test universe build..."
    $testJob.WaitForExit()
    if ($testJob.ExitCode -ne 0) { throw "test build failed - send reports/v3_test_build.err" }
} else {
    python "$src/predict_universe.py" --build-only 2>&1 | Tee-Object -FilePath reports/v3_test_build.log
    if ($LASTEXITCODE -ne 0) { throw "test build failed - send reports/v3_test_build.log" }
}

python "$src/predict_universe.py" --skip-build --tag base 2>&1 | Tee-Object -FilePath reports/v3_predict_console.log
if ($LASTEXITCODE -ne 0) { throw "prediction failed - send reports/v3_predict_console.log" }
python "$src/predict_universe.py" --skip-build --unseen-offset 0.05 --tag unseen05 2>&1 | Tee-Object -Append -FilePath reports/v3_predict_console.log
python "$src/make_v3_report.py"
Write-Host "DONE. Submit output/matching_results_base.tsv first. Send reports/v3_report_for_claude.md"
