"""
Herramienta de Análisis y Diagnóstico de Partidas de Kaggriculture.
Permite inspeccionar cualquier partida jugada en Kaggle:
- Puntuaciones finales y evolución de monedas día a día.
- Auditoría de mercado: compras y ventas totales por producto.
- Estado de la granja: animales, cosechas y jornaleros contratados.
- Detección de fallos y oportunidades de mejora.

Uso:
    python 06_kaggriculture/analyze_replay.py [EPISODE_ID]
"""

import json
import os
import subprocess
import sys
import tempfile

TEMP_DIR = tempfile.gettempdir()


def get_latest_submission_and_episodes():
    print("[*] Consultando últimos envíos en Kaggle...")
    res = subprocess.run(
        [r".venv\Scripts\kaggle.exe", "competitions", "submissions", "kaggriculture", "-v"],
        capture_output=True, text=True
    )
    lines = [l for l in res.stdout.strip().split("\n") if l.strip()]
    if len(lines) <= 1:
        print("[!] No se encontraron envíos.")
        return None, []

    # CSV headers: ref,totalBytes,date,description,status,publicScore,privateScore
    header = lines[0].split(",")
    latest_row = lines[1].split(",")
    ref_idx = header.index("ref") if "ref" in header else 0
    sub_id = latest_row[ref_idx]
    desc = latest_row[header.index("description")] if "description" in header else "Latest"

    print(f"[+] Último envío detectado: ID {sub_id} ({desc})")

    # Get episodes
    ep_res = subprocess.run(
        [r".venv\Scripts\kaggle.exe", "competitions", "episodes", sub_id, "-v"],
        capture_output=True, text=True
    )
    ep_lines = [l for l in ep_res.stdout.strip().split("\n") if l.strip()]
    episodes = []
    if len(ep_lines) > 1:
        ep_header = ep_lines[0].split(",")
        id_idx = ep_header.index("id") if "id" in ep_header else 0
        state_idx = ep_header.index("state") if "state" in ep_header else 3
        for row in ep_lines[1:]:
            parts = row.split(",")
            ep_id = parts[id_idx]
            ep_state = parts[state_idx] if len(parts) > state_idx else ""
            if "COMPLETED" in ep_state:
                episodes.append(ep_id)

    return sub_id, episodes


def download_replay(episode_id):
    replay_file = os.path.join(TEMP_DIR, f"episode-{episode_id}-replay.json")
    if os.path.exists(replay_file) and os.path.getsize(replay_file) > 1000:
        print(f"[+] Replay en caché local: {replay_file}")
        return replay_file

    print(f"[*] Descargando replay de la partida {episode_id}...")
    cmd = [
        r".venv\Scripts\kaggle.exe",
        "competitions",
        "replay",
        str(episode_id),
        "-p",
        TEMP_DIR,
    ]
    subprocess.run(cmd, check=True)
    return replay_file


def analyze(replay_path):
    print(f"\n{'='*70}")
    print(f"  ANÁLISIS DETALLADO DE PARTIDA: {os.path.basename(replay_path)}")
    print(f"{'='*70}\n")

    with open(replay_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    steps = data["steps"]
    total_steps = len(steps)
    final_step = steps[-1]

    # Scores
    score0 = final_step[0].get("reward", 0.0)
    score1 = final_step[1].get("reward", 0.0)

    winner = "Jugador 0" if score0 > score1 else ("Jugador 1" if score1 > score0 else "Empate")
    margin = abs(score0 - score1)

    print(f"[RESULTADO FINAL]")
    print(f"  * Jugador 0: ${score0:,.0f}")
    print(f"  * Jugador 1: ${score1:,.0f}")
    print(f"  * Ganador:   {winner} (Margen: ${margin:,.0f})\n")

    # Daily money curve
    print("-" * 70)
    print(" [EVOLUCIÓN DEL BANCO DÍA A DÍA]")
    print(f" {'Día':<6} | {'Jugador 0':<15} | {'Jugador 1':<15} | {'Líder':<10}")
    print("-" * 70)
    for d in [0, 5, 10, 15, 20, 25, 29]:
        step_idx = min(d * 24, total_steps - 1)
        m0 = steps[step_idx][0]["observation"]["farms"][0]["money"]
        m1 = steps[step_idx][1]["observation"]["farms"][1]["money"]
        leader = "P0" if m0 > m1 else ("P1" if m1 > m0 else "=")
        print(f" Día {d:02d} | ${m0:13,.0f} | ${m1:13,.0f} | {leader}")
    print("-" * 70 + "\n")

    # Market Audit (Buys and Sells)
    print("-" * 70)
    print(" [AUDITORÍA DE MERCADO: TRANSACCIONES TOTALES]")
    print("-" * 70)

    for pid in [0, 1]:
        sells = {}
        buys = {}
        hires = 0
        land_bought = 0

        for s in steps:
            action = s[pid].get("action", {})
            market = action.get("market", []) if isinstance(action, dict) else []
            for op in market:
                if not isinstance(op, list) or not op:
                    continue
                cmd = op[0]
                if cmd == "SELL" and len(op) >= 3:
                    item = op[1]
                    qty = int(op[2])
                    sells[item] = sells.get(item, 0) + qty
                elif cmd in ("BUY_PRODUCT", "BUY_SEED", "BUY_ANIMAL") and len(op) >= 3:
                    item = op[1]
                    qty = int(op[2])
                    buys[item] = buys.get(item, 0) + qty
                elif cmd == "HIRE":
                    hires += 1
                elif cmd == "BUY_LAND":
                    land_bought += 1

        print(f" >> Jugador {pid}:")
        print(f"    * Jornaleros contratados en total: {hires}")
        print(f"    * Expansiones de tierra (BUY_LAND): {land_bought}")
        print(f"    * Ventas totales:")
        for item, q in sorted(sells.items(), key=lambda x: -x[1]):
            print(f"        - {item:<12}: {q:>5} unidades vendidas")
        print(f"    * Compras totales:")
        for item, q in sorted(buys.items(), key=lambda x: -x[1]):
            print(f"        - {item:<12}: {q:>5} unidades compradas")

        # Warning detection: Wheat Churn
        wheat_bought = buys.get("WHEAT", 0)
        wheat_sold = sells.get("WHEAT", 0)
        if wheat_bought > 100 and wheat_sold > 100:
            churn = min(wheat_bought, wheat_sold)
            print(f"    [!] ALERTA: Detectado bucle de compra/venta de trigo (~{churn} unidades recicladas en vano).")
        print()

    # Final Farm State
    print("-" * 70)
    print(" [ESTADO DE LA GRANJA AL FINAL]")
    print("-" * 70)
    for pid in [0, 1]:
        farm = steps[-1][pid]["observation"]["farms"][pid]
        tiles = farm["tiles"]
        animals = [t for row in tiles for t in row if isinstance(t, dict) and "animal" in t]
        plants = [t for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "PLANT"]
        weeds = sum(1 for row in tiles for t in row if isinstance(t, dict) and t.get("kind") == "WEED")

        print(f" >> Jugador {pid}:")
        print(f"    * Cuadrantes desbloqueados: {farm.get('unlocked_quadrants')}")
        print(f"    * Animales vivos: {len(animals)} {[a['animal'] for a in animals]}")
        print(f"    * Plantas restantes: {len(plants)} {[p['crop'] for p in plants]}")
        print(f"    * Maleza acumulada: {weeds}")
        print()

    print(f"{'='*70}\n")


def main():
    if len(sys.argv) > 1:
        ep_id = sys.argv[1]
    else:
        sub_id, episodes = get_latest_submission_and_episodes()
        if not episodes:
            print("[!] No se encontraron episodios completados para analizar.")
            return
        ep_id = episodes[0]
        print(f"[+] Analizando el episodio más reciente: {ep_id}")

    replay_path = download_replay(ep_id)
    analyze(replay_path)


if __name__ == "__main__":
    main()
