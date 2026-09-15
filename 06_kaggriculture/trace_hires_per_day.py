import json
with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

hires_by_day = {}
for s in range(720):
    day = s // 24
    act = data['steps'][s][1].get('action', {})
    m_list = act.get('market', [])
    for m in m_list:
        if m and m[0] == 'HIRE':
            hires_by_day[day] = hires_by_day.get(day, 0) + 1

for d in range(30):
    print(f"Day {d:2d}: {hires_by_day.get(d, 0)} hands hired")
