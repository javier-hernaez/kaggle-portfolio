import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main

env = KaggricultureEnv(seed=42)

for s in range(720):
    obs0 = env.get_observation(0)
    a0 = main.agent(obs0)
    
    day = s // 24
    hour = s % 24
    if hour == 0 and day in [5, 8, 11, 15, 20]:
        tiles = obs0['farms'][0]['tiles']
        strawberries = sum(1 for r in tiles for c in r if isinstance(c, dict) and c.get('crop') == 'STRAWBERRY')
        melons = sum(1 for r in tiles for c in r if isinstance(c, dict) and c.get('crop') == 'MELON')
        wheat = sum(1 for r in tiles for c in r if isinstance(c, dict) and c.get('crop') == 'WHEAT')
        print(f"Day {day:2d}: Strawberries={strawberries}, Melons={melons}, Wheat={wheat}")
        
    env.step([a0, {}])
