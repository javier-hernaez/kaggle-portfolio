"""
Script de descarga y preparación de datos para la competición de Kaggle:
Name That LLM (https://www.kaggle.com/competitions/name-that-llm)
"""

import os
import shutil
import zipfile
import subprocess
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
COMPETITION_NAME = "name-that-llm"


def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    print("=" * 60)
    print(f" Descargando datos para: {COMPETITION_NAME}")
    print("=" * 60)

    # 1. Comprobar si ya existen los datos en scratch
    scratch_dir = os.path.join(os.path.dirname(BASE_DIR), "scratch", "name_that_llm")
    if os.path.exists(scratch_dir):
        copied = 0
        for fname in ["train.csv", "test.csv", "sample_submission.parquet"]:
            src = os.path.join(scratch_dir, fname)
            dst = os.path.join(DATA_DIR, fname)
            if os.path.exists(src) and (not os.path.exists(dst) or os.path.getsize(dst) == 0):
                shutil.copy2(src, dst)
                print(f"[+] Copiado desde scratch: {fname} ({os.path.getsize(dst):,} bytes)")
                copied += 1
        if copied > 0:
            print("[OK] Datos restaurados desde scratch.")
            return

    # 2. Intentar descargar mediante kaggle CLI / kagglehub
    try:
        import kagglehub
        print("[*] Descargando mediante kagglehub...")
        path = kagglehub.competition_download(COMPETITION_NAME)
        for fname in os.listdir(path):
            src = os.path.join(path, fname)
            dst = os.path.join(DATA_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"    -> Copiado {fname} ({os.path.getsize(dst):,} bytes)")
        print("\n[OK] ¡Datos descargados con éxito en data/!")
        return
    except Exception as e:
        print(f"[!] kagglehub fallo o no disponible: {e}")

    try:
        print("[*] Intentando descarga con Kaggle API...")
        cmd = [sys.executable, "-m", "kaggle", "competitions", "download", "-c", COMPETITION_NAME, "-p", DATA_DIR]
        subprocess.run(cmd, check=True)
        zip_path = os.path.join(DATA_DIR, f"{COMPETITION_NAME}.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
            print("[OK] ZIP descomprimido y eliminado.")
    except Exception as e:
        print(f"[!] Error con kaggle API: {e}")


if __name__ == "__main__":
    download_data()
