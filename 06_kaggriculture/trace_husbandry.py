import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main

env = KaggricultureEnv(seed=42)

for step in range(50):
    obs0 = env.get_observation(0)
    farmer_pos = obs0['farms'][0]['farmer']
    farmer_inv = obs0['private']['inventories'][0] if obs0['private'].get('inventories') else {}
    shed = obs0['private']['shed']
    
    a0 = main.agent(obs0)
    farmer_act = a0.get('farmer')
    
    # Check animal tiles
    tiles = obs0['farms'][0]['tiles']
    anim_tiles = []
    for y in range(10):
        for x in range(10):
            t = tiles[y][x]
            if isinstance(t, dict) and 'animal' in t:
                anim_tiles.append(((x, y), t['animal'], t.get('fed_today')))
                
    if step in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 24, 25, 26, 27, 28]:
        print(f"Step {step:2d}: Farmer@{farmer_pos} Inv={farmer_inv} Act={farmer_act} | Animals={anim_tiles} | ShedWheat={shed.get('WHEAT')}")
        
    env.step([a0, {}])
