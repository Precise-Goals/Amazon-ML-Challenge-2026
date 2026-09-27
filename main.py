"""
Google Colab Execution Script: Amazon ML Challenge 2026
Runs the unified full-universe training (India + US), test inference, and submission validation.
"""

import os
import sys
import subprocess
import time

def run_cmd(cmd, cwd=None):
    print(f"\n>>> Running: {cmd}", flush=True)
    res = subprocess.run(cmd, shell=True, cwd=cwd)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {res.returncode}: {cmd}")

def setup_environment():
    print("=== Step 1: Setting up environment ===", flush=True)
    run_cmd("pip install -q duckdb lightgbm rapidfuzz polars pyarrow joblib")

def setup_dataset():
    print("=== Step 2: Checking dataset location ===", flush=True)
    # Check standard Colab and local paths
    paths = [
        "datasource/dataset",
        "/content/datasource/dataset",
        "/content/drive/MyDrive/Amazon-ML-Challenge-2026/datasource/dataset",
        "/content/drive/MyDrive/dataset",
    ]
    for p in paths:
        if os.path.exists(os.path.join(p, "train", "train_source1.tsv")):
            print(f"Found dataset at: {p}", flush=True)
            return os.path.abspath(p)
            
    # Check if Google Drive is available
    if os.path.exists("/content/drive"):
        print("Checking Google Drive...", flush=True)
    else:
        try:
            from google.colab import drive
            print("Mounting Google Drive...", flush=True)
            drive.mount("/content/drive")
            for p in paths:
                if os.path.exists(os.path.join(p, "train", "train_source1.tsv")):
                    print(f"Found dataset in Google Drive at: {p}", flush=True)
                    return os.path.abspath(p)
        except Exception as e:
            print(f"Drive mount note: {e}", flush=True)
            
    # Check if a zip exists
    for zpath in ["dataset.zip", "/content/dataset.zip", "/content/drive/MyDrive/dataset.zip"]:
        if os.path.exists(zpath):
            print(f"Extracting {zpath}...", flush=True)
            run_cmd(f"unzip -q {zpath} -d datasource/dataset")
            return os.path.abspath("datasource/dataset")
            
    raise FileNotFoundError("Could not find dataset. Please mount Google Drive or upload dataset.zip to /content/.")

def main():
    t0 = time.time()
    print("=== Amazon ML Challenge 2026 — Colab Pipeline Runner ===", flush=True)
    setup_environment()
    data_dir = setup_dataset()
    
    src_dir = os.path.abspath("code/business_entity_resolution/src")
    sys.path.insert(0, src_dir)
    os.environ["PYTHONPATH"] = src_dir + ":" + os.environ.get("PYTHONPATH", "")
    
    print("\n=== Step 3: Running Unified Full-Universe Training (India + US) ===", flush=True)
    run_cmd(f"python -u {src_dir}/train_universe.py --data-dir {data_dir} --countries India US")
    
    print("\n=== Step 4: Running Test Prediction (Base) ===", flush=True)
    run_cmd(f"python -u {src_dir}/predict_universe.py --data-dir {data_dir} --tag base")
    
    print("\n=== Step 5: Running Fast Unseen Offset Rescore (Unseen05) ===", flush=True)
    run_cmd(f"python -u {src_dir}/predict_universe.py --data-dir {data_dir} --skip-build --unseen-offset 0.05 --tag unseen05")
    
    print("\n=== Step 6: Validating Submission ===", flush=True)
    val_script = "datasource/utils/validate_submission.py"
    if os.path.exists(val_script):
        run_cmd(f"python {val_script} --matching output/matching_results_base.tsv --candidate output/candidate_pairs_base.tsv --test-dir {data_dir}/test")
        run_cmd(f"python {val_script} --matching output/matching_results_unseen05.tsv --candidate output/candidate_pairs_unseen05.tsv --test-dir {data_dir}/test")
        
    print(f"\n=== COMPLETED in {(time.time()-t0)/60:.1f} minutes! ===", flush=True)
    print("Outputs available in: output/matching_results_base.tsv, output/matching_results_unseen05.tsv", flush=True)

if __name__ == "__main__":
    main()
