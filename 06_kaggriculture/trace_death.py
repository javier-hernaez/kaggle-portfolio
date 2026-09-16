import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main
from src.industrial_agent import industrial_agent

env = KaggricultureEnv(seed=2036)
prev_anims = 0

for s in range(720):
    obs0 = env.get_observation(0)
    obs1 = env.get_observation(1)
    a0 = main.agent(obs0)
    a1 = industrial_agent(obs1)
    env.step([a0, a1])
    
    day = s // 24
    hour = s % 24
    tiles = env.farms[0]['tiles']
    anims = sum(1 for r in tiles for c in r if isinstance(c, dict) and 'animal' in c)
    if hour == 0 or anims != prev_anims:
        if anims < prev_anims:
            print(f"--> Step {s:3d} (D{day:2d} H{hour:2d}): Animal DIED! Count went from {prev_anims} to {anims}")
        prev_anims = anims
