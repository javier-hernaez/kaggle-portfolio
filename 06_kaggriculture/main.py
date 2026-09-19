"""
Deep Grandmaster Agent (Industrial +1,000 Self-Play Evolved).
Chromosome Generation: Grandmaster_v10_Supreme

Evolved Parameters:
- NE Land: Day 5+ (Min Money: $1300)
- SW Land: Day 11+ (Min Money: $2500)
- Buy SE Land: False (Day 16, Min $4500)
- Animals: 10 Cows, 5 Sheep (Cutoff Day 12)
- Strawberries: Base 45 (Boost: +10 if IceCream/Smoothie shop opens)
- Sheep Synergy: +2 if Yarn Store opens
- Labor: Early 7, Mid 7 (Day 6), Late 10 (Day 10)
- Price Arbitrage: Throttle < 65%, Burst >= 110%
- Feed Reserve Mult: 1x
"""

CROPS = {
    "WHEAT":      {"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "ongoing": False},
    "CARROT":     {"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "ongoing": False},
    "TOMATO":     {"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "ongoing": True},
    "MELON":      {"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "ongoing": False},
}

BASE_PRICES = {
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100
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

SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}
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


class DeepChampionAgent:
    def __init__(self):
        pass

    def select_crop(self, day, remaining_days, money, strawberry_count, target_strawberries):
        if remaining_days <= 1:
            return None
        if remaining_days <= 4:
            if remaining_days >= 3 and money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if remaining_days >= 2 and money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 1: Melons + Wheat
        if day < 5:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 2: Strawberry Engine
        if 5 <= day <= 13:
            if strawberry_count < target_strawberries:
                return "STRAWBERRY"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 3: Wheat maintenance
        if day <= 25:
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 4: Quick carrots
        if 26 <= day <= 27:
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
        return None

    def plan_unit_turn(self, unit_idx, unit_pos, role, farm, private, day, hour, step, claimed_tiles, shared_state, target_strawberries):
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

        # --- 0. Endgame Drop ---
        if is_endgame and (carrying_sellable or carrying_animal):
            if (ux, uy) in SHED_ADJACENT:
                return ["DROP"], (ux, uy)
            d = _find_step((ux, uy), SHED_CENTER, board_size)
            return ([d] if d else ["PASS"]), SHED_CENTER

        # --- 1. Priority Shed Interaction ---
        if (ux, uy) in SHED_ADJACENT:
            if inv.get("FERTILIZER", 0) >= 3 or inv.get("MILK", 0) >= 3 or inv.get("WOOL", 0) >= 2 or inv.get("STRAWBERRY", 0) >= 4:
                return ["DROP"], (ux, uy)

            if role == "HUSBANDRY" and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                has_unfed = any(
                    isinstance(farm["tiles"][y][x], dict)
                    and "animal" in farm["tiles"][y][x]
                    and not farm["tiles"][y][x].get("fed_today", False)
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfed:
                    take_amt = min(4, shed.get("WHEAT", 0))
                    return ["PICKUP", "WHEAT", take_amt], (ux, uy)

            if role == "HUSBANDRY" and not carrying_animal and not shared_state["animal_pickup_claimed"]:
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

            if role == "AGRICULTURE" and day <= 6 and inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                has_unfert_melons = any(
                    isinstance(farm["tiles"][y][x], dict)
                    and farm["tiles"][y][x].get("crop") == "MELON"
                    and farm["tiles"][y][x].get("fertilized_until_day", -1) < day
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfert_melons:
                    return ["PICKUP", "FERTILIZER", 2], (ux, uy)

        # --- 2. Current Tile Actions ---
        if isinstance(tile, dict):
            if "animal" in tile:
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not tile.get("cared_today", False):
                    return ["CARE"], (ux, uy)
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"], (ux, uy)

            if tile.get("kind") == "PASTURE" and "animal" not in tile and carrying_animal:
                if carrying_cow:
                    return ["PLACE", "COW"], (ux, uy)
                if carrying_sheep:
                    return ["PLACE", "SHEEP"], (ux, uy)

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

            if tile.get("kind") == "WEED":
                return ["DIG"], (ux, uy)

        if (ux, uy) in PASTURE_LOCATIONS and _get_quadrant(ux, uy, board_size) in unlocked:
            if tile is None and (carrying_animal or role == "HUSBANDRY"):
                return ["BUILD_PASTURE"], (ux, uy)

        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in PASTURE_LOCATIONS:
            if _get_quadrant(ux, uy, board_size) in unlocked:
                best_crop = self.select_crop(day, remaining_days, farm["money"], shared_state.get("strawberries", 0), target_strawberries)
                if best_crop and seeds.get(best_crop, 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return ["PLANT", best_crop], (ux, uy)
                for alt_crop in ["STRAWBERRY", "MELON", "CARROT", "WHEAT"]:
                    if seeds.get(alt_crop, 0) > 0:
                        claimed_tiles.add((ux, uy))
                        return ["PLANT", alt_crop], (ux, uy)

        # --- 3. Target Selection by Role ---
        best_target = None
        best_prio = -1
        best_dist = 9999

        if role == "HUSBANDRY":
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

            for y in range(board_size):
                for x in range(board_size):
                    t = farm["tiles"][y][x]
                    if isinstance(t, dict) and "animal" in t:
                        if (x, y) in claimed_tiles:
                            continue
                        prio = -1
                        if t.get("yield_units", 0) > 0:
                            prio = 10
                        elif not t.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                            prio = 9
                        elif not t.get("cared_today", False):
                            prio = 8
                        elif t.get("fertilizer_available", False):
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

        else:
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
                        elif t.get("crop") == "MELON" and inv.get("FERTILIZER", 0) > 0 and t.get("fertilized_until_day", -1) < day:
                            prio = 7
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
        market_prices = obs.get("market", {}).get("prices", {})
        unlocked_shops = obs.get("town", {}).get("unlocked_shops", [])

        # Dynamic Town Shop Adaptation
        has_ice_cream_or_smoothie = any(s in ["ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "BRUNCH_SPOT"] for s in unlocked_shops)
        has_yarn_store = "YARN_STORE" in unlocked_shops

        eff_target_strawberries = 45 + (10 if has_ice_cream_or_smoothie else 0)
        eff_target_sheep = 5 + (2 if has_yarn_store else 0)

        market_orders = []

        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        total_sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_animals = total_cows + total_sheep
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)
        strawberry_count = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")

        # 1. Day 0 Master Opening
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

        # 2. Price-Aware Sales
        is_endgame_liquidation = (day >= 29 and hour >= 16) or (remaining_steps <= 12)
        wheat_reserve = (total_animals + animals_in_shed) * 1

        sell_schedule = [
            ("FERTILIZER", 12),
            ("MILK", 6),
            ("WOOL", 4),
            ("STRAWBERRY", 8),
            ("MELON", 6),
            ("WHEAT", 15),
            ("CARROT", 15),
            ("TOMATO", 10),
            ("EGG", 10),
        ]

        for prod, base_slice in sell_schedule:
            qty = shed.get(prod, 0)
            if qty <= 0 or len(market_orders) >= 9:
                continue

            curr_price = market_prices.get(prod, BASE_PRICES.get(prod, 50))
            base_p = BASE_PRICES.get(prod, 50)

            if not is_endgame_liquidation and curr_price < base_p * 0.65:
                effective_slice = max(1, base_slice // 2)
            elif curr_price >= base_p * 1.1:
                effective_slice = base_slice + 2
            else:
                effective_slice = base_slice

            if prod == "WHEAT" and not is_endgame_liquidation:
                sellable_wheat = max(0, qty - wheat_reserve)
                if sellable_wheat > 0:
                    market_orders.append(["SELL", "WHEAT", min(sellable_wheat, effective_slice)])
                continue

            sell_qty = qty if is_endgame_liquidation else min(qty, effective_slice)
            market_orders.append(["SELL", prod, sell_qty])

        # 3. Daily Labor Re-Hire
        hires_today = farm.get("hires_today", 0)
        if (hour == 1 or hour == 2) and remaining_days > 2:
            target_hires = 7
            if day >= 6:
                target_hires = 7
            if day >= 10:
                target_hires = 10

            while hires_today < target_hires and len(market_orders) < 9 and money >= 20:
                market_orders.append(["HIRE"])
                hires_today += 1
                money -= 20

        # 4. Gated Land Expansion
        if "NE" not in unlocked and money >= 1300 and day >= 5 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and day >= 11 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif False and "SE" not in unlocked and money >= 4500 and day >= 16 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 3000

        # 5. Balanced Animal Purchases
        total_capacity = len(PASTURE_LOCATIONS)
        if (total_animals + animals_in_shed) < total_capacity and day <= 12 and len(market_orders) < 9:
            if total_sheep < eff_target_sheep and (total_cows >= total_sheep * 2.5 or total_cows >= 10) and money >= 600:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            elif total_cows < 10 and money >= 500:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400

        # 6. Wheat Feed Procurement
        needed_feed = (total_animals + animals_in_shed) * 1
        current_wheat = shed.get("WHEAT", 0)
        if (total_animals + animals_in_shed) > 0 and current_wheat < needed_feed and money >= 100 and len(market_orders) < 9:
            buy_wheat_amt = min(8, needed_feed - current_wheat + 2)
            if buy_wheat_amt > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_amt])
                money -= buy_wheat_amt * 25

        # 7. Seed Purchases
        chosen_crop = self.select_crop(day, remaining_days, money, strawberry_count, eff_target_strawberries)
        if chosen_crop and remaining_days > 2 and len(market_orders) < 9:
            current_count = seeds.get(chosen_crop, 0)
            max_cap = 25 if chosen_crop == "STRAWBERRY" else 8
            if current_count < max_cap:
                seed_cost = CROPS[chosen_crop]["seed"]
                max_batch = 10 if chosen_crop == "STRAWBERRY" else 6
                buy_n = min(max_batch, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        # 8. Squad Specialization Execution
        claimed_tiles = set()
        shared_state = {
            "animal_pickup_claimed": False,
            "strawberries": strawberry_count
        }

        total_units = 1 + len(farm.get("hands", []))
        husbandry_count = 2 if total_animals <= 6 else 3

        farmer_role = "HUSBANDRY"
        farmer_act, _ = self.plan_unit_turn(0, farm["farmer"], farmer_role, farm, private, day, hour, step, claimed_tiles, shared_state, eff_target_strawberries)

        hands_acts = []
        for h_idx, h_pos in enumerate(farm.get("hands", [])):
            unit_id = h_idx + 1
            h_role = "HUSBANDRY" if unit_id < husbandry_count else "AGRICULTURE"
            h_act, _ = self.plan_unit_turn(unit_id, h_pos, h_role, farm, private, day, hour, step, claimed_tiles, shared_state, eff_target_strawberries)
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }


_champion_brain = DeepChampionAgent()


def agent(obs):
    """Kaggle Environments Entry Point."""
    return _champion_brain.act(obs)
