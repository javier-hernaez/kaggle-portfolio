import sys
sys.path.insert(0, '06_kaggriculture')
from mega_simulator.agent_genome import Genome, mutate
from mega_simulator.arena import Arena

if __name__ == '__main__':
    champion = Genome(name='Champion')
    challenger = mutate(champion)

    print('Testing Arena with 3 mirror seeds (6 games in parallel)...')
    arena = Arena(max_workers=4)
    res = arena.duel(challenger, champion, seeds=[42, 1039, 2036])
    print("Duel completed in {:.2f}s ({:.1f} games/sec)".format(res["elapsed_seconds"], res["games_per_second"]))
    print("Challenger: {} wins | Champion: {} wins | Ties: {}".format(res["wins_a"], res["wins_b"], res["ties"]))
    print("Avg scores: Challenger=${:,.0f} vs Champion=${:,.0f}".format(res["avg_score_a"], res["avg_score_b"]))
    print("Net Margin: ${:,.0f} | Challenger superior? {}".format(res["net_margin"], res["is_a_superior"]))
