import json
with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

# Look at all SELL actions
sell_orders = []
for s in range(720):
    act = data['steps'][s][1].get('action', {})
    for m in act.get('market', []):
        if m and m[0] == 'SELL' and m[1] == 'FERTILIZER':
            sell_orders.append((s, m))

print(f"Total SELL FERTILIZER order calls: {len(sell_orders)}")
print(f"Sum of requested quantities: {sum(m[2] if len(m) > 2 else 1 for s, m in sell_orders)}")

# But how many were ACTUALLY fulfilled?
# Look at shed inventory changes or observation!
total_fulfilled = 0
prev_shed_fert = 0
for s in range(720):
    obs = data['steps'][s][1]['observation']
    # Observation private shed
    shed = obs.get('private', {}).get('shed', {})
    # How much fertilizer was in shed over time?
