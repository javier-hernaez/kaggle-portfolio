import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main
from src.industrial_agent import industrial_agent

env = KaggricultureEnv(seed=42)

sales_by_item = {}
hires_total = 0

for step in range(720):
    obs0 = env.get_observation(0)
    obs1 = env.get_observation(1)
    a0 = main.agent(obs0)
    a1 = industrial_agent(obs1)
    
    for m in a0.get('market', []):
        if not m: continue
        if m[0] == 'SELL':
            item = m[1]
            qty = m[2] if len(m) > 2 else 1
            sales_by_item[item] = sales_by_item.get(item, 0) + qty
        elif m[0] == 'HIRE':
            hires_total += 1
            
    env.step([a0, a1])

print("=== OUR AGENT (P0) PERFORMANCE IN SEED 42 ===")
print(f"Final Score: ${env.farms[0]['money']:,.0f}")
print(f"Total Hires: {hires_total}")
print("Sales Breakdown:")
for k, v in sorted(sales_by_item.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k:15s}: {v:5d} units")

tiles0 = env.farms[0]['tiles']
animals0 = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and 'animal' in c)
pastures0 = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and c.get('kind') == 'PASTURE')
crops0 = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and c.get('kind') == 'PLANT')
print(f"Final board: {pastures0} Pastures, {animals0} Animals, {crops0} Crops")
