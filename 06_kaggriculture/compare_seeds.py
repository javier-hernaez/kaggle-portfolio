import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main
from src.industrial_agent import industrial_agent

def audit_seed(seed):
    env = KaggricultureEnv(seed=seed)
    sales = {}
    for s in range(720):
        obs0 = env.get_observation(0)
        obs1 = env.get_observation(1)
        a0 = main.agent(obs0)
        a1 = industrial_agent(obs1)
        for m in a0.get('market', []):
            if m and m[0] == 'SELL':
                item = m[1]
                qty = m[2] if len(m) > 2 else 1
                sales[item] = sales.get(item, 0) + qty
        env.step([a0, a1])
        
    tiles = env.farms[0]['tiles']
    anims = sum(1 for r in tiles for c in r if isinstance(c, dict) and 'animal' in c)
    pastures = sum(1 for r in tiles for c in r if isinstance(c, dict) and c.get('kind') == 'PASTURE')
    crops = sum(1 for r in tiles for c in r if isinstance(c, dict) and c.get('kind') == 'PLANT')
    money = env.farms[0]['money']
    print(f"\n=== SEED {seed}: Final Money = ${money:,.0f} ===")
    print(f"Pastures: {pastures}, Living Animals: {anims}, Final Crops: {crops}")
    print("Sales:", dict(sorted(sales.items(), key=lambda x: x[1], reverse=True)))
    print("Shops in Town:", env.town.get('unlocked_shops'))

audit_seed(2036) # The $39k game
audit_seed(9015) # The $92k game
