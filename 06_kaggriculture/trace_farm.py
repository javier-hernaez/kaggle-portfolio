import json

with open('episode-93364654-replay.json', 'r') as f:
    data = json.load(f)

print("=== EPISODE 93364654 (Majkel1337 - Elo 3285.3) FARM EVOLUTION ===")
for s in [1, 24, 72, 144, 240, 360, 480, 600, 719]:
    obs = data['steps'][s][1]['observation']
    pid = obs.get('player', 1)
    farm = obs['farms'][pid]
    day = obs['day']
    hour = obs['hour']
    coins = farm.get('coins')
    hands = len(farm.get('hands', []))
    animals = len(farm.get('animals', []))
    pastures = len(farm.get('pastures', []))
    plots = len(farm.get('plots', []))
    inventory = farm.get('inventory', {})
    print(f"Step {s:3d} (Day {day:2d}, H{hour:2d}): Coins=${coins:6,d} | Hands={hands:2d} | Animals={animals:2d} | Pastures={pastures:2d} | Crops={plots:2d} | Inv: {inventory}")
