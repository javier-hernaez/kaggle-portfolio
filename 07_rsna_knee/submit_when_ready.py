"""
Automated watcher that monitors kernel execution and submits to RSNA Knee competition when complete.
"""

import time
import subprocess
import sys
import shutil
from pathlib import Path

KERNEL = "javierhernaez/rsna-knee-baseline-submission"
COMPETITION = "rsna-knee-abnormality-detection"
VERSION = "8"
MESSAGE = "DINOsaur V5 v8: Rank-Logit Fusion + Clinical Comorbidity Lift + Repair-v1 CoAt Residual + Synovitis Rad Rescue (0.944+)"

venv_kaggle = Path(sys.executable).parent / ("kaggle.exe" if sys.platform == "win32" else "kaggle")
KAGGLE_BIN = str(venv_kaggle) if venv_kaggle.exists() else (shutil.which("kaggle") or "kaggle")


def check_status():
    cmd = [KAGGLE_BIN, "kernels", "status", KERNEL]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip()


def submit():
    cmd = [
        KAGGLE_BIN,
        "competitions",
        "submit",
        "-c",
        COMPETITION,
        "-k",
        KERNEL,
        "-f",
        "submission.csv",
        "-v",
        VERSION,
        "-m",
        MESSAGE,
    ]
    print(f"[*] Submitting version {VERSION} to competition...", flush=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("STDOUT:", res.stdout, flush=True)
    print("STDERR:", res.stderr, flush=True)
    return res.returncode == 0


def main():
    print(f"[*] Monitoring kernel: {KERNEL}", flush=True)
    max_wait = 14400  # 4 hours
    start = time.time()

    while time.time() - start < max_wait:
        status_line = check_status()
        print(f"[{int(time.time() - start)}s] {status_line}", flush=True)

        if "COMPLETE" in status_line:
            print("[+] Kernel completed successfully! Proceeding to submit...", flush=True)
            success = submit()
            if success:
                print("[+] Successfully submitted to competition!", flush=True)
            else:
                print("[-] Submission failed, see output above.", flush=True)
            return

        if "ERROR" in status_line or "CANCELLED" in status_line:
            print("[-] Kernel run ended with error or was cancelled.", flush=True)
            return

        time.sleep(60)


if __name__ == "__main__":
    main()
