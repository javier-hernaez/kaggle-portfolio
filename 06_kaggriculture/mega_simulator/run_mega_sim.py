import argparse
import sys
import os

sys.path.insert(0, os.path.abspath("06_kaggriculture"))

from mega_simulator.evolver import MegaEvolver
from mega_simulator.agent_genome import Genome
from mega_simulator.arena import Arena
from src.industrial_agent import industrial_agent
from src.heuristic_agent import heuristic_agent
from src.starter_baseline import starter_agent
from simulator.battle import tournament
import main


def run_benchmarks():
    print("=" * 70)
    print("  EJECUTANDO BENCHMARK HISTÓRICO DEL CAMPEÓN SUPREMO")
    print("=" * 70)

    print("\n[Test 1/3] Campeón vs Industrial v3 (6 partidas)...")
    tournament(main.agent, industrial_agent, num_games=6, agent0_name="Campeón_Supremo", agent1_name="Industrial_v3")

    print("\n[Test 2/3] Campeón vs Heuristic v1 (4 partidas)...")
    tournament(main.agent, heuristic_agent, num_games=4, agent0_name="Campeón_Supremo", agent1_name="Heuristic_v1")

    print("\n[Test 3/3] Campeón vs Starter Baseline (4 partidas)...")
    tournament(main.agent, starter_agent, num_games=4, agent0_name="Campeón_Supremo", agent1_name="Starter_Baseline")


def main_cli():
    parser = argparse.ArgumentParser(description="Mega-Simulador Industrial de Self-Play (+1,000 Partidas)")
    parser.add_argument("--total-games", "-g", type=int, default=1000, help="Total objetivo de partidas a simular")
    parser.add_argument("--pop-size", "-p", type=int, default=10, help="Tamaño de la población por generación")
    parser.add_argument("--seeds", "-s", type=int, default=5, help="Semillas por duelo espejo (x2 partidas)")
    parser.add_argument("--workers", "-w", type=int, default=6, help="Hilos paralelos de simulación")
    parser.add_argument("--benchmark", "-b", action="store_true", help="Ejecutar benchmark histórico")

    args = parser.parse_args()

    if args.benchmark:
        run_benchmarks()
        return

    evolver = MegaEvolver(workers=args.workers)
    evolver.run_mega_evolution(
        total_target_games=args.total_games,
        population_size=args.pop_size,
        seeds_per_duel=args.seeds
    )


if __name__ == "__main__":
    main_cli()
