import json

with open("episode-91174378-replay.json", 'r') as f:
    data = json.load(f)

print("=== SEB (ALLEGENDLY) (138,172 COINS) - DAY 0 STEP-BY-STEP ===")
for s in range(1, 15):
    act = data['steps'][s][1].get('action', {})
    print(f"Step {s:2d}: Farmer={act.get('farmer')} | Hands={act.get('hands')} | Market={act.get('market')}")
