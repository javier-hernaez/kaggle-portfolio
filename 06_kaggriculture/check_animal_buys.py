import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main
from src.industrial_agent import industrial_agent

def count_animal_buys(seed):
    env = KaggricultureEnv(seed=seed)
    cows_bought = 0
    sheep_bought = 0
    for s in range(720):
        obs0 = env.get_observation(0)
        obs1 = env.get_observation(1)
        a0 = main.agent(obs0)
        a1 = industrial_agent(obs1)
        for m in a0.get('market', []):
            if m and m[0] == 'BUY_ANIMAL':
                if m[1] == 'COW': cows_bought += (m[2] if len(m) > 2 else 1)
                if m[1] == 'SHEEP': sheep_bought += (m[2] if len(m) > 2 else 1)
        env.step([a0, a1])
    print(f"Seed {seed}: Bought {cows_bought} Cows, {sheep_bought} Sheep (Total: {cows_bought + sheep_bought})")

count_animal_buys(2036)
count_animal_buys(9015)
