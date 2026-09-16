import json
with open("episode-93364654-replay.json", 'r') as f:
    data = json.load(f)

money_by_product = {}
total_spent = 0
prev_money = 3000

for s in range(1, 720):
    obs = data['steps'][s][1]['observation']
    farm = obs['farms'][1]
    curr_money = farm['money']
    act = data['steps'][s][1].get('action', {})
    
    delta = curr_money - prev_money
    if delta > 0:
        # Check what was sold
        sold_items = [m[1] for m in act.get('market', []) if m and m[0] == 'SELL']
        # Attribute delta
        for item in sold_items:
            money_by_product[item] = money_by_product.get(item, 0) + delta / len(sold_items)
    elif delta < 0:
        total_spent += abs(delta)
        
    prev_money = curr_money

print(f"Final Reward: ${data['rewards'][1]:,.0f}")
print("Estimated Revenue by Product:")
for item, rev in sorted(money_by_product.items(), key=lambda x: x[1], reverse=True):
    print(f"  {item:15s}: ${rev:8,.0f}")
print(f"Total Money Spent (Animals, Seeds, Land, Labor): ${total_spent:,.0f}")
