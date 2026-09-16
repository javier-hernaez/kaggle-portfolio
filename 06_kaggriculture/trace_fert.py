import json
with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

fert_buys = 0
fert_sells = 0
fert_collects = 0

for s in range(720):
    act = data['steps'][s][1].get('action', {})
    m_list = act.get('market', [])
    for m in m_list:
        if m:
            if m[0] == 'BUY_PRODUCT' and len(m) > 1 and m[1] == 'FERTILIZER':
                fert_buys += m[2] if len(m) > 2 else 1
            if m[0] == 'SELL' and len(m) > 1 and m[1] == 'FERTILIZER':
                fert_sells += m[2] if len(m) > 2 else 1
                
    f_act = act.get('farmer', [])
    if f_act and f_act[0] == 'COLLECT_FERTILIZER':
        fert_collects += 1
    for h in act.get('hands', []):
        if h and h[0] == 'COLLECT_FERTILIZER':
            fert_collects += 1

print(f"MAJKEL1337 FERTILIZER STATS:")
print(f"  Bought from Market:    {fert_buys}")
print(f"  Collected from Animals:{fert_collects}")
print(f"  Sold to Market:        {fert_sells}")
