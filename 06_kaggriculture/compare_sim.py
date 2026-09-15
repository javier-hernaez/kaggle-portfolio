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
    env.step([a0, a1])
    
    if step % 72 == 0 or step in [1, 2, 5, 24, 719]:
        day = step // 24
        f0 = env.farms[0]
        f1 = env.farms[1]
        p0_p = sum(1 for r in f0['tiles'] for c in r if isinstance(c, dict) and c.get('kind') == 'PASTURE')
        p0_a = sum(1 for r in f0['tiles'] for c in r if isinstance(c, dict) and 'animal' in c)
        p1_p = sum(1 for r in f1['tiles'] for c in r if isinstance(c, dict) and c.get('kind') == 'PASTURE')
        p1_a = sum(1 for r in f1['tiles'] for c in r if isinstance(c, dict) and 'animal' in c)
        s0 = env.privates[0]['shed']
        s1 = env.privates[1]['shed']
        print(f"Step {step:3d} (D{day:2d}): P0(v5) Money=${f0['money']:6,.0f}, Hands={len(f0['hands'])}, Past={p0_p}, Anim={p0_a} | Shed={dict(s0)}")
        print(f"               P1(v3) Money=${f1['money']:6,.0f}, Hands={len(f1['hands'])}, Past={p1_p}, Anim={p1_a} | Shed={dict(s1)}")
