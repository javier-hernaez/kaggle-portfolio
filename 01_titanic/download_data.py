"""
Data download script for Titanic Competition.
Tries kagglehub first; if unauthenticated, downloads directly into data/
"""

import os
import shutil

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_data():
    print("Intentando descargar con kagglehub...")
    try:
        import kagglehub
        path = kagglehub.competition_download('titanic')
        print(f"Descargado con éxito mediante kagglehub en: {path}")
        
        # Copiar archivos a la carpeta local data/
        for fname in os.listdir(path):
            src = os.path.join(path, fname)
            dst = os.path.join(DATA_DIR, fname)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        print("Archivos copiados a la carpeta local 'data/'.")
        return
    except Exception as e:
        print(f"Aviso con kagglehub ({e}).")
        print("Descargando directamente los archivos oficiales del dataset...")

    import urllib.request
    base_url = "https://raw.githubusercontent.com/logpresso/dataset/main/titanic/"
    files = ["train.csv", "test.csv", "gender_submission.csv"]
    for f in files:
        target = os.path.join(DATA_DIR, f)
        print(f"Descargando {f}...")
        urllib.request.urlretrieve(base_url + f, target)
        print(f"-> {f} listo ({os.path.getsize(target)} bytes).")

    print("\n¡Todos los datos están listos en la carpeta 'data/'!")

if __name__ == "__main__":
    download_data()
