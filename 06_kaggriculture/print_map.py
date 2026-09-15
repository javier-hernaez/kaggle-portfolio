import json

with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

obs = data['steps'][24][1]['observation']
farm = obs['farms'][1]
tiles = farm['tiles']

print("=== MAJKEL1337 TILE MAP AT DAY 1 ===")
for y in range(10):
    row_str = ""
    for x in range(10):
        c = tiles[y][x]
        if c == 'LOCKED':
            row_str += " . "
        elif c is None:
            row_str += " _ "
        elif isinstance(c, dict):
            if 'animal' in c:
                row_str += f"A{c['animal'][0]} "
            elif c.get('kind') == 'PASTURE':
                row_str += " P "
            elif c.get('kind') == 'PLANT':
                row_str += f"{c['crop'][0].lower()}  "
            else:
                row_str += " ? "
        else:
            row_str += " ! "
    print(f"Y={y}: {row_str}")

obs15 = data['steps'][360][1]['observation']
tiles15 = obs15['farms'][1]['tiles']
print("\n=== MAJKEL1337 TILE MAP AT DAY 15 ===")
for y in range(10):
    row_str = ""
    for x in range(10):
        c = tiles15[y][x]
        if c == 'LOCKED':
            row_str += " . "
        elif c is None:
            row_str += " _ "
        elif isinstance(c, dict):
            if 'animal' in c:
                row_str += f"A{c['animal'][0]} "
            elif c.get('kind') == 'PASTURE':
                row_str += " P "
            elif c.get('kind') == 'PLANT':
                row_str += f"{c['crop'][0].lower()}  "
            else:
                row_str += " ? "
        else:
            row_str += " ! "
    print(f"Y={y}: {row_str}")
