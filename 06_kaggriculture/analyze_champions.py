import json
import urllib.request
import os

episodes_to_analyze = [93364654, 91174378, 90960991]

def download_and_analyze(ep_id):
    filename = f"replay_{ep_id}.json"
    url = f"https://www.kaggle.com/api/i/competitions.EpisodeService/ShowEpisode"
    # Or use kaggle cli
    print(f"\n==========================================")
    print(f"ANALYZING EPISODE {ep_id}")
    print(f"==========================================")
    
    # Download via Kaggle CLI
    import subprocess
    cmd = [r".venv\Scripts\kaggle.exe", "competitions", "replay", str(ep_id)]
    subprocess.run(cmd, capture_output=True)
    
    rep_path = f"episode-{ep_id}-replay.json"
    if not os.path.exists(rep_path):
        print(f"Failed to find {rep_path}")
        return
        
    with open(rep_path, 'r') as f:
        data = json.load(f)
        
    os.remove(rep_path) # Clean up large file
    
    # Identify players and winning player
    rewards = data.get('rewards', [])
    print(f"Final Rewards: {rewards}")
    win_idx = 0 if rewards[0] > rewards[1] else 1
    print(f"Winner is Player {win_idx} with {rewards[win_idx]:,.0f} coins (vs {rewards[1-win_idx]:,.0f})")
    
    steps = data.get('steps', [])
    print(f"Total steps: {len(steps)}")
    
    # Let's inspect Winner's:
    # 1. Day 0 actions (step 0 to 24)
    # 2. Worker hires over time
    # 3. Land purchases over time
    # 4. Animal purchases & structures
    # 5. Crop planting breakdown (which crops, when, how many)
    # 6. Market transaction stats
    
    hires = []
    land_buys = []
    market_buys = {}
    market_sells = {}
    crop_counts = {}
    structures = set()
    daily_balance = {}
    
    for s_idx, step_pair in enumerate(steps):
        day = s_idx // 24
        p_step = step_pair[win_idx]
        obs = p_step.get('observation', {})
        act = p_step.get('action', {})
        
        if s_idx % 24 == 0:
            daily_balance[day] = obs.get('coins', 0)
            
        farm = obs.get('farm', {})
        
        if not act:
            continue
            
        m_actions = act.get('market_actions', [])
        for m in m_actions:
            m_type = m.get('type')
            item = m.get('item')
            qty = m.get('quantity', 1)
            if m_type == 'BUY':
                market_buys[item] = market_buys.get(item, 0) + qty
                if 'LAND' in item:
                    land_buys.append((s_idx, day, item))
            elif m_type == 'SELL':
                market_sells[item] = market_sells.get(item, 0) + qty
            elif m_type == 'HIRE_HAND':
                hires.append((s_idx, day))
                
        w_actions = act.get('worker_actions', {})
        # Track planting or building
        for wid, w_act in w_actions.items():
            a_type = w_act.get('type')
            if a_type == 'PLANT':
                crop = w_act.get('crop')
                crop_counts[crop] = crop_counts.get(crop, 0) + 1
            elif a_type == 'BUILD':
                st = w_act.get('structure')
                structures.add(st)
                
    print(f"\n--- TIMELINE OF WORKER HIRES ---")
    print(f"Total hands hired: {len(hires)}")
    hire_by_day = {}
    for s, d in hires:
        hire_by_day[d] = hire_by_day.get(d, 0) + 1
    for d, c in sorted(hire_by_day.items()):
        print(f"  Day {d:2d}: {c} hands")
        
    print(f"\n--- TIMELINE OF LAND PURCHASES ---")
    for s, d, item in land_buys:
        print(f"  Day {d:2d} (step {s}): {item}")
        
    print(f"\n--- MARKET PURCHASES ---")
    for item, qty in sorted(market_buys.items(), key=lambda x: x[1], reverse=True):
        print(f"  {item:20s}: {qty}")
        
    print(f"\n--- CROPS PLANTED BREAKDOWN ---")
    for crop, qty in sorted(crop_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {crop:20s}: {qty} plants")
        
    print(f"\n--- MARKET SALES BREAKDOWN ---")
    for item, qty in sorted(market_sells.items(), key=lambda x: x[1], reverse=True):
        print(f"  {item:20s}: {qty}")
        
    print(f"\n--- BALANCE AT START OF DAY (Sample) ---")
    for d in range(0, 30, 3):
        print(f"  Day {d:2d}: ${daily_balance.get(d, 0):,}")
    print(f"  Day 29 end reward: ${rewards[win_idx]:,.0f}")

for ep in episodes_to_analyze:
    download_and_analyze(ep)
