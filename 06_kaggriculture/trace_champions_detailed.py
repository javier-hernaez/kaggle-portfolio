import json

def trace(ep_id, title):
    with open(f"episode-{ep_id}-replay.json", 'r') as f:
        data = json.load(f)
    print(f"\n=================================================================")
    print(f"=== {title} (Episode {ep_id}) ===")
    print(f"=================================================================")
    rewards = data.get('rewards', [0, 0])
    win_idx = 0 if rewards[0] > rewards[1] else 1
    
    for s in [1, 24, 72, 144, 240, 360, 480, 600, 719]:
        p_step = data['steps'][s][win_idx]
        obs = p_step['observation']
        farm = obs['farms'][win_idx]
        day = obs.get('day', s // 24)
        hour = obs.get('hour', s % 24)
        money = farm.get('money', 0)
        hands = len(farm.get('hands', []))
        quads = farm.get('unlocked_quadrants', [])
        
        tiles = farm.get('tiles', [])
        t_types = {}
        for row in tiles:
            for cell in row:
                if not cell:
                    continue
                if isinstance(cell, str):
                    t_types[cell] = t_types.get(cell, 0) + 1
                elif isinstance(cell, dict):
                    k = cell.get('kind', 'UNKNOWN')
                    crop = cell.get('crop')
                    struct = cell.get('structure')
                    name = f"{k}_{crop}" if crop else f"{k}_{struct}" if struct else k
                    t_types[name] = t_types.get(name, 0) + 1
                    
        print(f"Step {s:3d} (Day {day:2d}, H{hour:2d}): Money=${money:8,.1f} | Hands={hands:2d} | Quads={len(quads)} | Tiles: {dict(t_types)}")

trace(93364654, "Majkel1337 (Elo 3285.3, Top 1 Mundial)")
trace(91174378, "Seb (allegedly) (Record Historico 138,172 Monedas)")
