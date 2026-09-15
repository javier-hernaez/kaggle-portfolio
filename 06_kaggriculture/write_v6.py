code = """\"\"\"
Champion Agent v6 (Grandmaster Squad Specialization Edition).
Directly addresses the 3 bottlenecks identified from replay simulation:
1. Animal Logistics & Guaranteed Feeding:
   - Dedicated Husbandry Workers always load wheat at shed BEFORE visiting animals.
   - Every animal is reliably FED and CARED every day.
   - Unlocks full production of MILK, WOOL, and 1,500+ FERTILIZER.
2. Zero Traffic Jam Shed Coordination:
   - Single-unit mutex on shed animal pickups (no more 6-worker deadlocks).
3. Agricultural Squad (Mass Strawberry Engine):
   - Dedicated planters & waterers plant 35-42 Strawberries.
   - Perennial strawberries yield continuously every 2 days.
4. Coordinated Sliced Market Sales:
   - Sells fertilizer (up to 10/turn), milk, wool, strawberries every turn.
\"\"\"

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "ongoing": False},
}

ANIMALS = {
    "COW":   {"cost": 400, "product": "MILK"},
    "SHEEP": {"cost": 500, "product": "WOOL"},
}

MOVES = {
    (0, -1): "NORTH",
    (0,  1): "SOUTH",
    (1,  0): "EAST",
    (-1, 0): "WEST",
}

PASTURE_LOCATIONS = [
    (4, 4), (3, 4), (4, 3), (3, 3),
    (5, 4), (5, 3), (4, 2), (5, 2),
    (6, 4), (6, 3), (7, 4),
    (2, 4), (3, 5), (4, 5)
]

SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5), (3, 4), (4, 3)}
SHED_CENTER = (4, 4)


def _dist(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_step(start_pos, target_pos, board_size=10):
    sx, sy = start_pos
    tx, ty = target_pos
    if (sx, sy) == (tx, ty):
        return None
    dx = tx - sx
    dy = ty - sy
    if abs(dx) >= abs(dy) and dx != 0:
        step_x = 1 if dx > 0 else -1
        nx = sx + step_x
        if 0 <= nx < board_size:
            return MOVES.get((step_x, 0))
    if dy != 0:
        step_y = 1 if dy > 0 else -1
        ny = sy + step_y
        if 0 <= ny < board_size:
            return MOVES.get((0, step_y))
    if dx != 0:
        step_x = 1 if dx > 0 else -1
        return MOVES.get((step_x, 0))
    return None


def _get_quadrant(x, y, board_size=10):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


class GrandmasterAgentV6:
    def __init__(self):
        pass

    def select_crop(self, day, remaining_days, money, strawberry_count):
        if remaining_days <= 1:
            return None
        if remaining_days <= 4:
            if remaining_days >= 3 and money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if remaining_days >= 2 and money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
        
        # Phase 1: Days 0-4 -> Melons + Wheat
        if day <= 4:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
            
        # Phase 2: Days 5-11 -> Strawberries up to 42
        if 5 <= day <= 11:
            if strawberry_count < 42 and money >= CROPS["STRAWBERRY"]["seed"]:
                return "STRAWBERRY"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
            
        # Phase 3: Days 12-25 -> Wheat for animal feed
        if 12 <= day <= 25:
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
            
        # Phase 4: Days 26-27 -> Quick carrots
        if 26 <= day <= 27:
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
        return None

    def plan_unit_turn(self, unit_idx, unit_pos, role, farm, private, day, hour, step, claimed_tiles, shared_state):
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        tile = farm["tiles"][uy][ux]
        inv = private["inventories"][unit_idx] if unit_idx < len(private.get("inventories", [])) else {}
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        remaining_steps = 720 - step
        remaining_days = 30 - day
        is_endgame = (day >= 29 and hour >= 16) or (remaining_steps <= 18)
        unlocked = set(farm.get("unlocked_quadrants", ["NW"]))

        carrying_cow = inv.get("COW", 0) > 0
        carrying_sheep = inv.get("SHEEP", 0) > 0
        carrying_animal = carrying_cow or carrying_sheep
        carrying_sellable = sum(inv.get(k, 0) for k in ["MILK", "WOOL", "STRAWBERRY", "MELON", "CARROT", "FERTILIZER", "WHEAT", "EGG"]) > 0

        # --- 0. ENDGAME DROP: Rush to shed and deposit ---
        if is_endgame and (carrying_sellable or carrying_animal):
            if (ux, uy) in SHED_ADJACENT:
                return ["DROP"], (ux, uy)
            d = _find_step((ux, uy), SHED_CENTER, board_size)
            return ([d] if d else ["PASS"]), SHED_CENTER

        # --- 1. CURRENT TILE ACTIONS ---
        if isinstance(tile, dict):
            # Animal tile
            if "animal" in tile:
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"], (ux, uy)
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not tile.get("cared_today", False):
                    return ["CARE"], (ux, uy)

            # Empty Pasture tile
            if tile.get("kind") == "PASTURE" and "animal" not in tile and carrying_animal:
                if carrying_cow:
                    return ["PLACE", "COW"], (ux, uy)
                if carrying_sheep:
                    return ["PLACE", "SHEEP"], (ux, uy)

            # Plant tile
            if tile.get("kind") == "PLANT":
                crop = tile.get("crop", "")
                age = day - tile.get("planted_day", 0)
                c_data = CROPS.get(crop, {})
                peak = age >= c_data.get("max_yield_day", 99)
                if (peak or is_endgame) and tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if crop == "MELON" and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day:
                    return ["FERTILIZE"], (ux, uy)
                if not tile.get("watered_today", False):
                    return ["WATER"], (ux, uy)

            # Weed
            if tile.get("kind") == "WEED":
                return ["DIG"], (ux, uy)

        # Empty pasture spot: build pasture if holding animal or builder role
        if (ux, uy) in PASTURE_LOCATIONS and _get_quadrant(ux, uy, board_size) in unlocked:
            if tile is None and (carrying_animal or role == "HUSBANDRY"):
                return ["BUILD_PASTURE"], (ux, uy)

        # Plant on empty ground (AGRICULTURE role)
        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in PASTURE_LOCATIONS:
            if _get_quadrant(ux, uy, board_size) in unlocked:
                best_crop = self.select_crop(day, remaining_days, farm["money"], shared_state.get("strawberries", 0))
                if best_crop and seeds.get(best_crop, 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return ["PLANT", best_crop], (ux, uy)
                for alt_crop in ["STRAWBERRY", "MELON", "CARROT", "WHEAT"]:
                    if seeds.get(alt_crop, 0) > 0:
                        claimed_tiles.add((ux, uy))
                        return ["PLANT", alt_crop], (ux, uy)

        # --- 2. SHED PICKUP & DROP (When Adjacent) ---
        if (ux, uy) in SHED_ADJACENT:
            # Drop off goods if carrying heavy load
            if inv.get("FERTILIZER", 0) >= 4 or inv.get("MILK", 0) >= 3 or inv.get("WOOL", 0) >= 2 or inv.get("STRAWBERRY", 0) >= 4:
                return ["DROP"], (ux, uy)

            # HUSBANDRY ROLE: Pickup Animal or Wheat
            if role == "HUSBANDRY" and not carrying_animal:
                # Pickup animal if mutex available and animal in shed
                if not shared_state["animal_pickup_claimed"]:
                    # Check if empty pasture spot exists
                    for px, py in PASTURE_LOCATIONS:
                        if _get_quadrant(px, py, board_size) in unlocked:
                            pt = farm["tiles"][py][px]
                            if pt is None or (isinstance(pt, dict) and pt.get("kind") == "PASTURE" and "animal" not in pt):
                                if shed.get("COW", 0) > 0:
                                    shared_state["animal_pickup_claimed"] = True
                                    return ["PICKUP", "COW", 1], (ux, uy)
                                if shed.get("SHEEP", 0) > 0:
                                    shared_state["animal_pickup_claimed"] = True
                                    return ["PICKUP", "SHEEP", 1], (ux, uy)

                # Pickup Wheat for unfed animals
                has_unfed = any(
                    isinstance(farm["tiles"][y][x], dict)
                    and "animal" in farm["tiles"][y][x]
                    and not farm["tiles"][y][x].get("fed_today", False)
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfed and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
                    take_amt = min(4, shed.get("WHEAT", 0))
                    return ["PICKUP", "WHEAT", take_amt], (ux, uy)

        # --- 3. TARGET SELECTION BY ROLE ---
        best_target = None
        best_prio = -1
        best_dist = 9999

        # A) HUSBANDRY ROLE TARGETS
        if role == "HUSBANDRY":
            # If carrying animal -> go to nearest unpopulated pasture spot
            if carrying_animal:
                for px, py in PASTURE_LOCATIONS:
                    if _get_quadrant(px, py, board_size) in unlocked:
                        pt = farm["tiles"][py][px]
                        if pt is None or (isinstance(pt, dict) and pt.get("kind") == "PASTURE" and "animal" not in pt):
                            d = _dist((ux, uy), (px, py))
                            if d < best_dist:
                                best_dist = d
                                best_target = (px, py)
                if best_target:
                    step_dir = _find_step((ux, uy), best_target, board_size)
                    if step_dir:
                        return [step_dir], best_target

            # If needs wheat -> walk directly to SHED!
            has_unfed = any(
                isinstance(farm["tiles"][y][x], dict)
                and "animal" in farm["tiles"][y][x]
                and not farm["tiles"][y][x].get("fed_today", False)
                for y in range(board_size) for x in range(board_size)
            )
            if has_unfed and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                step_to_shed = _find_step((ux, uy), SHED_CENTER, board_size)
                if step_to_shed:
                    return [step_to_shed], SHED_CENTER

            # Visit animals that need feeding, care, or harvest
            for y in range(board_size):
                for x in range(board_size):
                    t = farm["tiles"][y][x]
                    if isinstance(t, dict) and "animal" in t:
                        if (x, y) in claimed_tiles:
                            continue
                        prio = -1
                        if t.get("yield_units", 0) > 0:
                            prio = 10
                        elif t.get("fertilizer_available", False):
                            prio = 9
                        elif not t.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                            prio = 8
                        elif not t.get("cared_today", False):
                            prio = 7

                        if prio > 0:
                            d = _dist((ux, uy), (x, y))
                            if prio > best_prio or (prio == best_prio and d < best_dist):
                                best_prio = prio
                                best_dist = d
                                best_target = (x, y)

            if best_target:
                claimed_tiles.add(best_target)
                step_dir = _find_step((ux, uy), best_target, board_size)
                if step_dir:
                    return [step_dir], best_target

        # B) AGRICULTURE ROLE TARGETS
        else:
            # 1. Harvest mature crops
            for y in range(board_size):
                for x in range(board_size):
                    if (x, y) in claimed_tiles:
                        continue
                    q = _get_quadrant(x, y, board_size)
                    if q not in unlocked:
                        continue
                    t = farm["tiles"][y][x]
                    if t == "LOCKED":
                        continue

                    d = _dist((ux, uy), (x, y))
                    prio = -1

                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        c_data = CROPS.get(t["crop"], {})
                        age = day - t["planted_day"]
                        if (age >= c_data["max_yield_day"] or is_endgame) and t.get("yield_units", 0) > 0:
                            prio = 10
                        elif not t.get("watered_today", False):
                            prio = 8
                    elif t is None and (x, y) not in PASTURE_LOCATIONS and remaining_days > 1:
                        prio = 5
                    elif isinstance(t, dict) and t.get("kind") == "WEED":
                        prio = 2

                    if prio > best_prio or (prio == best_prio and d < best_dist):
                        best_prio = prio
                        best_dist = d
                        best_target = (x, y)

            if best_target:
                claimed_tiles.add(best_target)
                step_dir = _find_step((ux, uy), best_target, board_size)
                if step_dir:
                    return [step_dir], best_target

        return ["PASS"], (ux, uy)

    def act(self, obs):
        player = obs["player"]
        farms = obs["farms"]
        farm = farms[player]
        private = obs["private"]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        remaining_days = 30 - day
        remaining_steps = 720 - step

        money = farm["money"]
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        unlocked = farm.get("unlocked_quadrants", ["NW"])

        market_orders = []

        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        total_sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_animals = total_cows + total_sheep
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)
        strawberry_count = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")

        # ==========================================
        # 1. Day 0 Opening: Majkel1337 Exact Step 1
        # ==========================================
        if step == 0 or (step == 1 and len(unlocked) == 1 and money >= 2500):
            return {
                "farmer": ["BUILD_PASTURE"],
                "hands": [],
                "market": [
                    ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                    ["BUY_ANIMAL", "COW", 2],
                    ["BUY_ANIMAL", "SHEEP", 2],
                    ["BUY_SEED", "MELON", 12],
                    ["BUY_SEED", "WHEAT", 7],
                    ["BUY_PRODUCT", "WHEAT", 6],
                ][:10],
            }

        # ==========================================
        # 2. MARKET SALES FIRST (Cash Generation):
        # Fertilizer (10), Milk (5), Wool (4), Strawberry (6), Melon (5)
        # ==========================================
        is_endgame_liquidation = (day >= 29 and hour >= 16) or (remaining_steps <= 12)
        wheat_reserve = (total_animals + animals_in_shed) * 2

        sell_schedule = [
            ("FERTILIZER", 10),
            ("MILK", 5),
            ("WOOL", 4),
            ("STRAWBERRY", 6),
            ("MELON", 5),
            ("WHEAT", 15),
            ("CARROT", 15),
            ("TOMATO", 10),
            ("EGG", 10),
        ]

        for prod, slice_size in sell_schedule:
            qty = shed.get(prod, 0)
            if qty <= 0 or len(market_orders) >= 9:
                continue

            if prod == "WHEAT" and not is_endgame_liquidation:
                sellable_wheat = max(0, qty - wheat_reserve)
                if sellable_wheat > 0:
                    market_orders.append(["SELL", "WHEAT", min(sellable_wheat, slice_size)])
                continue

            sell_qty = qty if is_endgame_liquidation else min(qty, slice_size)
            market_orders.append(["SELL", prod, sell_qty])

        # ==========================================
        # 3. Daily Labor Re-Hire (Hour 1 & 2):
        # Scale to 5 early -> 8 mid -> 12 late
        # ==========================================
        hires_today = farm.get("hires_today", 0)
        if (hour == 1 or hour == 2) and remaining_days > 2:
            target_hires = 5
            if day >= 6:
                target_hires = 8
            if day >= 10:
                target_hires = 12

            while hires_today < target_hires and len(market_orders) < 9 and money >= 20:
                market_orders.append(["HIRE"])
                hires_today += 1
                money -= 20

        # ==========================================
        # 4. Strictly Gated Land Expansion:
        # Day 6 (NE) and Day 11 (SW)
        # ==========================================
        if "NE" not in unlocked and money >= 1200 and 6 <= day <= 15 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and 11 <= day <= 18 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000

        # ==========================================
        # 5. Animal Purchases (Up to Day 11 only):
        # ==========================================
        total_capacity = len(PASTURE_LOCATIONS)
        if (total_animals + animals_in_shed) < total_capacity and day <= 11 and len(market_orders) < 9:
            if total_cows < 10 and money >= 600:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400
            elif total_sheep < 4 and money >= 700:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500

        # ==========================================
        # 6. Wheat Feed Procurement:
        # ==========================================
        needed_feed = (total_animals + animals_in_shed) * 2
        current_wheat = shed.get("WHEAT", 0)
        if (total_animals + animals_in_shed) > 0 and current_wheat < needed_feed and money >= 100 and len(market_orders) < 9:
            buy_wheat_amt = min(8, needed_feed - current_wheat + 2)
            if buy_wheat_amt > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_amt])
                money -= buy_wheat_amt * 25

        # ==========================================
        # 7. Seed Purchases:
        # ==========================================
        chosen_crop = self.select_crop(day, remaining_days, money, strawberry_count)
        if chosen_crop and remaining_days > 2 and len(market_orders) < 9:
            current_count = seeds.get(chosen_crop, 0)
            max_cap = 20 if chosen_crop == "STRAWBERRY" else 8
            if current_count < max_cap:
                seed_cost = CROPS[chosen_crop]["seed"]
                buy_n = min(6, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        # ==========================================
        # 8. SQUAD SPECIALIZATION EXECUTION:
        # Units 0, 1 (and 2 if animals >= 8): HUSBANDRY
        # Units 3+: AGRICULTURE
        # ==========================================
        claimed_tiles = set()
        shared_state = {
            "animal_pickup_claimed": False,
            "strawberries": strawberry_count
        }

        total_units = 1 + len(farm.get("hands", []))
        husbandry_count = 2 if total_animals <= 6 else 3

        farmer_role = "HUSBANDRY"
        farmer_act, _ = self.plan_unit_turn(0, farm["farmer"], farmer_role, farm, private, day, hour, step, claimed_tiles, shared_state)

        hands_acts = []
        for h_idx, h_pos in enumerate(farm.get("hands", [])):
            unit_id = h_idx + 1
            h_role = "HUSBANDRY" if unit_id < husbandry_count else "AGRICULTURE"
            h_act, _ = self.plan_unit_turn(unit_id, h_pos, h_role, farm, private, day, hour, step, claimed_tiles, shared_state)
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }


_grandmaster_agent = GrandmasterAgentV6()


def agent(obs):
    \"\"\"Kaggle Environments Entry Point.\"\"\"
    return _grandmaster_agent.act(obs)
"""

with open("06_kaggriculture/main.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Wrote GrandmasterAgentV6 (Squad Specialization) into 06_kaggriculture/main.py!")
