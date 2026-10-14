"""
Evaluation and Tournament Benchmark Suite for Kaggriculture.
Evaluates the production Grandmaster Agent against baseline agents and mirror duels.

Usage:
    python evaluate.py [--vs {all,mirror,industrial,heuristic,starter}] [--games N] [--verbose]
"""

import os
import sys
import argparse
from typing import Dict, Any, List

# Add project root to sys.path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from simulator.battle import run_match, tournament
from main import agent as grandmaster_agent
from src.industrial_agent import industrial_agent
from src.heuristic_agent import heuristic_agent
from src.starter_baseline import starter_agent, random_agent


def run_balanced_duel(agent0, agent1, name0: str, name1: str, seeds: List[int], verbose: bool = False) -> Dict[str, Any]:
    """Runs a balanced duel where both agents interchange P0 and P1 across all seeds to cancel turn bias."""
    wins0 = 0
    wins1 = 0
    draws = 0
    scores0 = []
    scores1 = []
    
    for s in seeds:
        # Game 1: Agent0 is P0
        res1 = run_match(agent0, agent1, seed=s)
        s0, s1 = res1["score0"], res1["score1"]
        scores0.append(s0)
        scores1.append(s1)
        if s0 > s1:
            wins0 += 1
        elif s1 > s0:
            wins1 += 1
        else:
            draws += 1
            
        if verbose:
            print(f"  [Seed {s:4d} - G1] {name0} (P0): ${s0:,.0f} vs {name1} (P1): ${s1:,.0f}")

        # Game 2: Agent0 is P1 (balanced swap)
        res2 = run_match(agent1, agent0, seed=s)
        s1_alt, s0_alt = res2["score0"], res2["score1"]
        scores0.append(s0_alt)
        scores1.append(s1_alt)
        if s0_alt > s1_alt:
            wins0 += 1
        elif s1_alt > s0_alt:
            wins1 += 1
        else:
            draws += 1

        if verbose:
            print(f"  [Seed {s:4d} - G2] {name0} (P1): ${s0_alt:,.0f} vs {name1} (P0): ${s1_alt:,.0f}")

    total_games = len(scores0)
    avg0 = sum(scores0) / total_games
    avg1 = sum(scores1) / total_games
    winrate0 = (wins0 / total_games) * 100
    net_margin = avg0 - avg1
    max0 = max(scores0)

    return {
        "name0": name0,
        "name1": name1,
        "games": total_games,
        "wins0": wins0,
        "wins1": wins1,
        "draws": draws,
        "winrate0": winrate0,
        "avg0": avg0,
        "avg1": avg1,
        "net_margin": net_margin,
        "max0": max0,
    }


def main():
    parser = argparse.ArgumentParser(description="Kaggriculture Agent Benchmark Suite")
    parser.add_argument("--vs", choices=["all", "mirror", "industrial", "heuristic", "starter"], default="all",
                        help="Opponent category to benchmark against")
    parser.add_argument("--games", type=int, default=6, help="Number of distinct seeds per matchup (balanced duels run 2x)")
    parser.add_argument("--verbose", action="store_true", help="Print step details per match")
    args = parser.parse_args()

    default_seeds = [42, 1039, 2036, 3033, 4030, 5027, 6024, 7021, 8018, 9015][:args.games]

    print("\n" + "=" * 75)
    print("  KAGGRICULTURE GRANDMASTER EVALUATION BENCHMARK")
    print(f"  Configuration: {len(default_seeds)} seeds ({len(default_seeds)*2} balanced games per matchup)")
    print("=" * 75)

    results = []

    # 1. Industrial Agent v3
    if args.vs in ["all", "industrial"]:
        print("\n[1/3] Benchmarking vs Industrial Agent v3...")
        r = run_balanced_duel(grandmaster_agent, industrial_agent, "Grandmaster_v13", "Industrial_v3", default_seeds, args.verbose)
        results.append(r)
        print(f"  => Winrate: {r['winrate0']:.1f}% ({r['wins0']}W/{r['wins1']}L) | Avg: ${r['avg0']:,.0f} vs ${r['avg1']:,.0f} (Margin: +${r['net_margin']:,.0f})")

    # 2. Heuristic Agent v1
    if args.vs in ["all", "heuristic"]:
        print("\n[2/3] Benchmarking vs Heuristic Agent v1...")
        r = run_balanced_duel(grandmaster_agent, heuristic_agent, "Grandmaster_v13", "Heuristic_v1", default_seeds, args.verbose)
        results.append(r)
        print(f"  => Winrate: {r['winrate0']:.1f}% ({r['wins0']}W/{r['wins1']}L) | Avg: ${r['avg0']:,.0f} vs ${r['avg1']:,.0f} (Margin: +${r['net_margin']:,.0f})")

    # 3. Starter Baseline
    if args.vs in ["all", "starter"]:
        print("\n[3/3] Benchmarking vs Official Starter Baseline...")
        r = run_balanced_duel(grandmaster_agent, starter_agent, "Grandmaster_v13", "Starter_Baseline", default_seeds, args.verbose)
        results.append(r)
        print(f"  => Winrate: {r['winrate0']:.1f}% ({r['wins0']}W/{r['wins1']}L) | Avg: ${r['avg0']:,.0f} vs ${r['avg1']:,.0f} (Margin: +${r['net_margin']:,.0f})")

    # 4. Mirror Match (Self-Play check)
    if args.vs in ["mirror"]:
        print("\n[*] Running Self-Play Mirror Matches...")
        r = run_balanced_duel(grandmaster_agent, grandmaster_agent, "Grandmaster_P0", "Grandmaster_P1", default_seeds, args.verbose)
        results.append(r)
        print(f"  => Self-Play Parity: Avg P0 ${r['avg0']:,.0f} vs Avg P1 ${r['avg1']:,.0f}")

    # Summary Table
    print("\n" + "=" * 75)
    print("  TOURNAMENT SUMMARY TABLE")
    print("=" * 75)
    print(f"{'Opponent':<20} | {'Games':<5} | {'Winrate':<8} | {'Avg Score':<12} | {'Rival Score':<12} | {'Net Margin':<10}")
    print("-" * 75)
    for r in results:
        print(f"{r['name1']:<20} | {r['games']:<5} | {r['winrate0']:>6.1f}% | ${r['avg0']:>10,.0f} | ${r['avg1']:>10,.0f} | +${r['net_margin']:>8,.0f}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
