"""
Evaluation and Tournament Benchmark for Kaggriculture.
Compares Heuristic Agent vs Starter Agent and Random Baseline.
"""

import os
import sys

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(__file__))

from simulator.battle import tournament
from src.starter_baseline import starter_agent, random_agent
from src.heuristic_agent import heuristic_agent


def main():
    print("\n" + "=" * 65)
    print("  INICIANDO BENCHMARK DE SIMULACION - KAGGRICULTURE")
    print("=" * 65)

    # 1. Benchmark: Heuristic Agent vs Official Starter Agent
    print("\n[Torneo 1] Heuristic Agent vs Official Starter Baseline...")
    res_starter = tournament(
        heuristic_agent,
        starter_agent,
        num_games=6,
        start_seed=100,
        agent0_name="HeuristicAgent_v1",
        agent1_name="StarterBaseline",
    )

    # 2. Benchmark: Heuristic Agent vs Random Agent
    print("\n[Torneo 2] Heuristic Agent vs Random Agent...")
    res_random = tournament(
        heuristic_agent,
        random_agent,
        num_games=4,
        start_seed=200,
        agent0_name="HeuristicAgent_v1",
        agent1_name="RandomAgent",
    )

    print("\n" + "=" * 65)
    print("  RESUMEN GLOBAL DEL BENCHMARK:")
    print(f"  * Vs Starter Baseline: Winrate {res_starter['winrate0']:.1f}% | Media ${res_starter['avg0']:,.0f} vs ${res_starter['avg1']:,.0f}")
    print(f"  * Vs Random Baseline:  Winrate {res_random['winrate0']:.1f}% | Media ${res_random['avg0']:,.0f} vs ${res_random['avg1']:,.0f}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
