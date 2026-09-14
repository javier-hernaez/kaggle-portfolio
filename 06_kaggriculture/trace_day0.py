import json

with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

print("=== MAJKEL1337 (Elo 3285.3, 118,513 coins) - DAY 0 STEP-BY-STEP ===")
for s in range(1, 15):
    act = data['steps'][s][1].get('action', {})
    print(f"Step {s:2d}: Farmer={act.get('farmer')} | Hands={act.get('hands')} | Market={act.get('market')}")
