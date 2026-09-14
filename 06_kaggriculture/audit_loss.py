import json

cmd = [r".venv\Scripts\kaggle.exe", "competitions", "replay", "108971739"]
import subprocess
subprocess.run(cmd, capture_output=True)

with open("episode-108971739-replay.json", 'r') as f:
    data = json.load(f)

# P1 is our agent (7434 coins), P0 is opponent (103715 coins)
print("=== AUDIT OF EPISODE 108971739 ===")
print("Rewards:", data['rewards'])
for s in [1, 2, 5, 24, 72, 144, 240, 480, 719]:
    step = data['steps'][s]
    # Check our agent (P1)
    obs1 = step[1]['observation']
    farm1 = obs1['farms'][1]
    act1 = step[1].get('action', {})
    
    # Check opponent (P0)
    obs0 = step[0]['observation']
    farm0 = obs0['farms'][0]
    
    print(f"Step {s:3d} (D{obs1['day']:2d} H{obs1['hour']:2d}):")
    print(f"   OURS (P1): Money=${farm1['money']:6,.1f}, Hands={len(farm1['hands'])}, Farmer={act1.get('farmer')}, Market={act1.get('market')[:3]}")
    print(f"   OPP  (P0): Money=${farm0['money']:6,.1f}, Hands={len(farm0['hands'])}")

