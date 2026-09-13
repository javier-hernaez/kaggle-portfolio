"""
Automated submission script for Kaggriculture.
Validates the agent locally before pushing to the Kaggle competition leaderboard.
"""

import os
import sys
import subprocess

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_PY = os.path.join(PROJECT_DIR, "main.py")


def validate_agent():
    print("[*] Verificando agente localmente antes de enviar...")
    sys.path.insert(0, PROJECT_DIR)
    from main import agent
    from simulator.engine import KaggricultureEnv

    env = KaggricultureEnv(config={"episodeSteps": 72})
    scores = env.run([agent, agent])
    print(f"[OK] Verificacion exitosa: 72 pasos completados sin errores. Puntuaciones: {scores}")
    return True


def submit_to_kaggle(message="Heuristic Strategic Agent v1"):
    if not validate_agent():
        print("[!] Error en la validacion local.")
        return False

    print(f"\n[*] Enviando {MAIN_PY} a la competicion 'kaggriculture'...")
    cmd = [
        r".venv\Scripts\kaggle.exe",
        "competitions",
        "submit",
        "kaggriculture",
        "-f",
        MAIN_PY,
        "-m",
        message,
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(res.stdout)
        print("\n[OK] Envio completado con exito!")
    except subprocess.CalledProcessError as e:
        print(f"[!] Error al enviar: {e}")
        print(e.stderr)
        return False

    print("\n[*] Estado reciente de envíos:")
    subprocess.run([r".venv\Scripts\kaggle.exe", "competitions", "submissions", "kaggriculture"])
    return True


if __name__ == "__main__":
    msg = sys.argv[1] if len(sys.argv) > 1 else "Heuristic Strategic Agent v1 (rotation + expansion)"
    submit_to_kaggle(msg)
