import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main
from src.industrial_agent import industrial_agent

env = KaggricultureEnv(seed=42)

for step in range(720):
    obs0 = env.get_observation(0)
    obs1 = env.get_observation(1)
    a0 = main.agent(obs0)
    a1 = industrial_agent(obs1)
    
    day = step // 24
    hour = step % 24
    
    # Check feeding actions
    feed_count = 0
    if a0.get('farmer') and a0['farmer'][0] == 'FEED':
        feed_count += 1
    for h in a0.get('hands', []):
        if h and h[0] == 'FEED':
            feed_count += 1
            
    env.step([a0, a1])
    
    if hour == 23:
        # End of day check
        tiles0 = env.farms[0]['tiles']
        total_anim = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and 'animal' in c)
        unfed = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and 'animal' in c and not c.get('fed_today', False))
        shed_wheat = env.privates[0]['shed'].get('WHEAT', 0)
        crops = sum(1 for r in tiles0 for c in r if isinstance(c, dict) and c.get('kind') == 'PLANT')
        print(f"Day {day:2d} End: Animals={total_anim} (Unfed={unfed}), Wheat in shed={shed_wheat}, Crops={crops}")
