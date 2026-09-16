with open("06_kaggriculture/main.py", "r", encoding="utf-8") as f:
    code = f.read()

# Make strawberry the absolute priority in Days 5-13
old_select = """        # Phase 2: Days 5-11 -> Strawberries up to 42
        if 5 <= day <= 11:
            if strawberry_count < 42 and money >= CROPS["STRAWBERRY"]["seed"]:
                return "STRAWBERRY"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None"""

new_select = """        # Phase 2: Days 5-13 -> STRAWBERRY SUPER-ENGINE
        if 5 <= day <= 13:
            if strawberry_count < 40:
                return "STRAWBERRY"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None"""

assert old_select in code, "old_select not found!"
code = code.replace(old_select, new_select)

# Also in Section 7 (Seed Purchases), allow buying up to 10 strawberries at once
old_buy = """            max_cap = 20 if chosen_crop == "STRAWBERRY" else 8
            if current_count < max_cap:
                seed_cost = CROPS[chosen_crop]["seed"]
                buy_n = min(6, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost"""

new_buy = """            max_cap = 25 if chosen_crop == "STRAWBERRY" else 8
            if current_count < max_cap:
                seed_cost = CROPS[chosen_crop]["seed"]
                max_batch = 10 if chosen_crop == "STRAWBERRY" else 6
                buy_n = min(max_batch, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost"""

assert old_buy in code, "old_buy not found!"
code = code.replace(old_buy, new_buy)

with open("06_kaggriculture/main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Updated Strawberry priority and batch sizes!")
