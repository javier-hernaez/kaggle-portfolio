import sys
sys.path.insert(0, '06_kaggriculture')
from simulator.engine import KaggricultureEnv
import main

env = KaggricultureEnv(seed=42)

for step in range(30):
    obs0 = env.get_observation(0)
    obs1 = env.get_observation(1)
    a0 = main.agent(obs0)
    
    farmer_act = a0.get('farmer')
    hands_act = a0.get('hands')
    market_act = a0.get('market')
    
    print(f"Step {step:2d} (H{step%24:2d}): Farmer={farmer_act} | Hands={hands_act}")
    if market_act:
        print(f"         Market={market_act}")
        
    env.step([a0, {}])
