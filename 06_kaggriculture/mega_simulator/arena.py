import copy
import time
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Dict, Tuple

from simulator.engine import KaggricultureEnv
from mega_simulator.agent_genome import Genome, make_agent


def _run_single_match_worker(args):
    """Worker function executed in separate process."""
    dict_a, dict_b, seed, flip = args
    genome_a = Genome.from_dict(dict_a)
    genome_b = Genome.from_dict(dict_b)

    agent_a = make_agent(genome_a)
    agent_b = make_agent(genome_b)

    # If flip is False: P0 = A, P1 = B
    # If flip is True:  P0 = B, P1 = A
    env = KaggricultureEnv(seed=seed)
    p0_agent = agent_b if flip else agent_a
    p1_agent = agent_a if flip else agent_b

    scores = env.run([p0_agent, p1_agent])
    score_a = scores[1] if flip else scores[0]
    score_b = scores[0] if flip else scores[1]

    return {
        "seed": seed,
        "flip": flip,
        "score_a": score_a,
        "score_b": score_b,
        "margin_a": score_a - score_b,
        "win_a": 1.0 if score_a > score_b else (0.5 if score_a == score_b else 0.0),
    }


class Arena:
    """Multiprocessing Arena for Mirror-Match Head-to-Head Tournaments."""
    def __init__(self, max_workers: int = 6):
        self.max_workers = max_workers

    def duel(self, genome_a: Genome, genome_b: Genome, seeds: List[int]) -> Dict:
        """
        Runs full mirror matches across all seeds.
        For each seed: (A vs B) and (B vs A). Total matches = 2 * len(seeds).
        """
        t0 = time.perf_counter()
        tasks = []
        dict_a = genome_a.to_dict()
        dict_b = genome_b.to_dict()

        for s in seeds:
            tasks.append((dict_a, dict_b, s, False))
            tasks.append((dict_a, dict_b, s, True))

        results = []
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            for res in executor.map(_run_single_match_worker, tasks):
                results.append(res)

        elapsed = time.perf_counter() - t0

        total_games = len(results)
        wins_a = sum(1 for r in results if r["win_a"] == 1.0)
        ties = sum(1 for r in results if r["win_a"] == 0.5)
        wins_b = sum(1 for r in results if r["win_a"] == 0.0)
        avg_score_a = sum(r["score_a"] for r in results) / total_games
        avg_score_b = sum(r["score_b"] for r in results) / total_games
        net_margin = sum(r["margin_a"] for r in results) / total_games
        winrate_a = (wins_a + 0.5 * ties) / total_games

        return {
            "total_games": total_games,
            "wins_a": wins_a,
            "wins_b": wins_b,
            "ties": ties,
            "winrate_a": winrate_a,
            "avg_score_a": avg_score_a,
            "avg_score_b": avg_score_b,
            "net_margin": net_margin,
            "elapsed_seconds": elapsed,
            "games_per_second": total_games / max(0.001, elapsed),
            "is_a_superior": (winrate_a >= 0.55 and net_margin > 0),
        }
