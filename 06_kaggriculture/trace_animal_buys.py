import json
with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

for s in range(720):
    act = data['steps'][s][1].get('action', {})
    m_list = act.get('market', [])
    for m in m_list:
        if m and m[0] == 'BUY_ANIMAL':
            print(f"Step {s:3d} (Day {s//24:2d}): {m}")
