"""
Battle & Tournament runner for Kaggriculture.
Allows benchmarking agents against baselines with statistics and detailed reports.
"""

import time
from .engine import KaggricultureEnv


def run_match(agent0, agent1, seed=None, config=None, verbose=False):
    """
    Runs a single 720-step match between agent0 and agent1.
    Returns:
        dict with {
            "score0": float,
            "score1": float,
            "winner": 0 | 1 | "TIE",
            "margin": float,
            "elapsed": float
        }
    """
    t0 = time.perf_counter()
    env = KaggricultureEnv(config=config, seed=seed)
    scores = env.run([agent0, agent1])
    elapsed = time.perf_counter() - t0

    score0, score1 = scores[0], scores[1]
    if score0 > score1:
        winner = 0
    elif score1 > score0:
        winner = 1
    else:
        winner = "TIE"

    margin = abs(score0 - score1)
    if verbose:
        print(f"[Match] P0: ${score0:,.0f} vs P1: ${score1:,.0f} | Winner: P{winner} (Margin: ${margin:,.0f}) | Time: {elapsed:.2f}s")

    return {
        "score0": score0,
        "score1": score1,
        "winner": winner,
        "margin": margin,
        "elapsed": elapsed,
    }


def tournament(agent0, agent1, num_games=10, start_seed=42, agent0_name="Agent0", agent1_name="Agent1"):
    """
    Runs a round-robin tournament alternating player positions.
    """
    print("=" * 65)
    print(f"  TOURNAMENT: {agent0_name} vs {agent1_name} ({num_games} partidas)")
    print("=" * 65)

    wins0 = 0
    wins1 = 0
    ties = 0
    scores0 = []
    scores1 = []
    t_start = time.perf_counter()

    for g in range(num_games):
        seed = start_seed + g * 997
        # Alternate sides for fair benchmarking
        if g % 2 == 0:
            res = run_match(agent0, agent1, seed=seed)
            s0, s1 = res["score0"], res["score1"]
            scores0.append(s0)
            scores1.append(s1)
            w = agent0_name if res["winner"] == 0 else (agent1_name if res["winner"] == 1 else "Empate")
        else:
            res = run_match(agent1, agent0, seed=seed)
            s0, s1 = res["score1"], res["score0"]
            scores0.append(s0)
            scores1.append(s1)
            w = agent1_name if res["winner"] == 0 else (agent0_name if res["winner"] == 1 else "Empate")

        if s0 > s1:
            wins0 += 1
        elif s1 > s0:
            wins1 += 1
        else:
            ties += 1

        print(f" Partida {g+1:02d}/{num_games:02d} [Seed {seed:6d}]: {agent0_name} ${s0:,.0f} vs {agent1_name} ${s1:,.0f} -> {w}")

    total_time = time.perf_counter() - t_start
    avg0 = sum(scores0) / len(scores0)
    avg1 = sum(scores1) / len(scores1)
    winrate0 = (wins0 / num_games) * 100

    print("-" * 65)
    print(f" RESUMEN FINAL:")
    print(f"  * {agent0_name}: {wins0} victorias ({winrate0:.1f}%) | Media: ${avg0:,.1f}")
    print(f"  * {agent1_name}: {wins1} victorias ({(wins1/num_games)*100:.1f}%) | Media: ${avg1:,.1f}")
    if ties:
        print(f"  * Empates: {ties}")
    print(f"  * Tiempo total: {total_time:.2f}s ({total_time/num_games:.3f}s/partida)")
    print("=" * 65)

    return {
        "wins0": wins0,
        "wins1": wins1,
        "ties": ties,
        "avg0": avg0,
        "avg1": avg1,
        "winrate0": winrate0,
    }
