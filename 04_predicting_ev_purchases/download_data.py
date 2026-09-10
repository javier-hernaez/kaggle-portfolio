"""
Script de descarga para la competición Playground Series Season 6 Episode 9:
Predicting Electric Vehicle Purchases (Will_Buy_EV)
"""

import os
import shutil
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_data():
    print("=" * 60)
    print(" Descargando datos de Playground Series s6e9...")
    print("=" * 60)
    
    # Intento 1: Usando kagglehub
    try:
        import kagglehub
        print("[*] Conectando mediante kagglehub...")
        path = kagglehub.competition_download('playground-series-s6e9')
        print(f"[+] Archivos descargados en caché: {path}")
        
        for fname in os.listdir(path):
            src = os.path.join(path, fname)
            dst = os.path.join(DATA_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
                print(f"    -> Copiado {fname} ({os.path.getsize(dst):,} bytes)")
        print("\n[✔] ¡Todos los archivos están listos en 'data/'!")
        return
    except Exception as e:
        print(f"[!] kagglehub aviso: {e}")
        
    # Intento 2: Usando kaggle CLI
    try:
        print("[*] Intentando mediante kaggle CLI...")
        os.system(f'kaggle competitions download -c playground-series-s6e9 -p "{DATA_DIR}"')
        zip_path = os.path.join(DATA_DIR, "playground-series-s6e9.zip")
        if os.path.exists(zip_path):
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATA_DIR)
            os.remove(zip_path)
            print("[✔] Archivo zip extraído y eliminado con éxito.")
            return
    except Exception as e:
        print(f"[!] Error con kaggle CLI: {e}")

if __name__ == "__main__":
    download_data()
