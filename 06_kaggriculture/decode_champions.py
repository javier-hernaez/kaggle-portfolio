import json
import os

episodes = [93364654, 91174378]

for ep_id in episodes:
    rep_path = f"episode-{ep_id}-replay.json"
    if not os.path.exists(rep_path):
        import subprocess
        cmd = [r".venv\Scripts\kaggle.exe", "competitions", "replay", str(ep_id)]
        subprocess.run(cmd, capture_output=True)
        
    with open(rep_path, 'r') as f:
        data = json.load(f)
        
    rewards = data.get('rewards', [])
    win_idx = 0 if rewards[0] > rewards[1] else 1
    winner_name = data.get('info', {}).get('TeamNames', ['P0', 'P1'])[win_idx]
    safe_winner = str(winner_name).encode('ascii', 'ignore').decode()
    print(f"\n=================================================================")
    print(f"EPISODE {ep_id}: {safe_winner} (Player {win_idx}) WINNER with {rewards[win_idx]:,.0f} coins")
    print(f"=================================================================")
    
    steps = data.get('steps', [])
    
    market_events = {}
    land_events = []
    farmer_actions = {}
    hands_actions = {}
    hand_counts = []
    day_balances = {}
    
    for s_idx, step_pair in enumerate(steps):
        day = s_idx // 24
        p_step = step_pair[win_idx]
        obs = p_step.get('observation', {})
        act = p_step.get('action') or {}
        
        if s_idx % 24 == 0:
            day_balances[day] = obs.get('coins')
            
        m_list = act.get('market', [])
        for m in m_list:
            if not m: continue
            cmd = m[0]
            if cmd == 'HIRE':
                market_events['HIRE'] = market_events.get('HIRE', 0) + 1
            elif cmd in ('BUY_LAND', 'BUY_PROPERTY'):
                land_events.append((s_idx, day, m))
            elif cmd in ('BUY_ANIMAL', 'BUY_SEED', 'BUY_PRODUCT'):
                item = m[1] if len(m) > 1 else '?'
                qty = m[2] if len(m) > 2 else 1
                key = f"{cmd}_{item}"
                market_events[key] = market_events.get(key, 0) + qty
            elif cmd == 'SELL':
                item = m[1] if len(m) > 1 else '?'
                qty = m[2] if len(m) > 2 else 1
                key = f"SELL_{item}"
                market_events[key] = market_events.get(key, 0) + qty
                
        f_act = act.get('farmer')
        if f_act:
            cmd = f_act[0]
            farmer_actions[cmd] = farmer_actions.get(cmd, 0) + 1
            
        h_acts = act.get('hands', [])
        for h in h_acts:
            if h:
                cmd = h[0]
                hands_actions[cmd] = hands_actions.get(cmd, 0) + 1
                
        # Farmer position and pasture positions
        if s_idx in [1, 2, 3, 4, 5, 24, 48, 120, 240, 480, 719]:
            farm = obs.get('farm', {})
            farmer = farm.get('farmer', {})
            hands = farm.get('hands', [])
            structures = farm.get('structures', {})
            animals = farm.get('animals', [])
            plots = farm.get('plots', {})
            print(f"Step {s_idx:3d} (Day {day:2d}): Coins=${obs.get('coins')}, Hands={len(hands)}, Animals={len(animals)}, Structures={len(structures)}, Crops={len(plots)}")
            if s_idx in [1, 2, 3, 4, 5]:
                print(f"   -> Act: farmer={act.get('farmer')}, market={act.get('market')}")
                
    print("\n--- TOTAL MARKET ACTIONS ---")
    for k, v in sorted(market_events.items()):
        print(f"  {k:25s}: {v}")
        
    print("\n--- LAND PURCHASES ---")
    for s, d, m in land_events:
        print(f"  Day {d:2d} (step {s:3d}): {m}")
        
    print("\n--- FARMER ACTIONS SUMMARY ---")
    for k, v in sorted(farmer_actions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:20s}: {v}")
        
    print("\n--- HANDS ACTIONS SUMMARY ---")
    for k, v in sorted(hands_actions.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k:20s}: {v}")
        
    print("\n--- DAILY BALANCE ---")
    for d in range(0, 30, 3):
        val = day_balances.get(d)
        print(f"  Day {d:2d}: ${val:,}" if val is not None else f"  Day {d:2d}: N/A")
    print(f"  Day 29 End: ${rewards[win_idx]:,.0f}")
