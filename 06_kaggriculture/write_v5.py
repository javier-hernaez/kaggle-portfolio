code = """\"\"\"
Champion Agent v5 (Grandmaster Edition).
Reverse-engineered directly from the #1 player in the world:
- Majkel1337 (Elo 3285.3, 118,513 coins in Episode 93364654)
- Seb (138,172 coins all-time record in Episode 91174378)

CORE WINNING PILLARS:
1. Daily Labor Scaling (Hands expire every 24h at midnight! Re-hire 5-14 hands every morning at Hour 1).
2. Day 0 Zero-Land Rapid Livestock Deployment (Farmer builds pasture step 1; all 4 animals placed by step 5).
3. Delayed Land Expansion (NE on Day 6+, SW on Day 10+, SE on Day 15+; never starve early cash).
4. Strawberry Engine Transition (Days 6-20: mass planting of perennial strawberries generating $4k+/day).
5. Fertilizer Gold Mine (Collect & sell animal manure daily; up to 1,500 units sold).
6. Daily Feed & Care Guarantee (Protected wheat feed reserve for cows and sheep).
7. Endgame Rush & Full Liquidation (Steps 714-720: drop all carried items at shed and sell 100% of inventory).
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
    (3, 4), (4, 3), (3, 3), (3, 5), (5, 3),
    (2, 3), (2, 4), (2, 5), (3, 2), (4, 2),
    (1, 3), (1, 4), (2, 2), (3, 1), (4, 1),
    (6, 3), (6, 4), (5, 2), (6, 2)
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


class ChampionAgentV5:
    def __init__(self):
        pass

    def select_crop(self, day, remaining_days, money):
        if remaining_days <= 1:
            return None
        if remaining_days <= 4:
            if remaining_days >= 3 and money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if remaining_days >= 2 and money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
        
        # Phase 1 (Days 0-5): Melons for massive initial burst + Wheat for animal feed
        if day <= 5:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None
            
        # Phase 2 (Days 6-20): Strawberry Super-Engine (reharvestable indefinitely)
        if day <= 20:
            if money >= CROPS["STRAWBERRY"]["seed"]:
                return "STRAWBERRY"
            if money >= CROPS["MELON"]["seed"] and day <= 12:
                return "MELON"
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            return None
            
        # Phase 3 (Days 21-27): Quick Carrots before the season ends
        if day <= 27:
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
        return None

    def plan_unit(self, unit_idx, unit_pos, farm, private, day, hour, step, claimed_tiles):
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        tile = farm["tiles"][uy][ux]
        inv = private["inventories"][unit_idx] if unit_idx < len(private.get("inventories", [])) else {}
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        remaining_steps = 720 - step
        remaining_days = 30 - day
        is_endgame = remaining_steps <= 18 or (day == 29 and hour >= 18)
        unlocked = set(farm.get("unlocked_quadrants", ["NW"]))

        carrying_cow = inv.get("COW", 0) > 0
        carrying_sheep = inv.get("SHEEP", 0) > 0
        carrying_animal = carrying_cow or carrying_sheep
        carrying_sellable = sum(inv.get(k, 0) for k in ["MILK", "WOOL", "STRAWBERRY", "MELON", "CARROT", "FERTILIZER", "WHEAT", "EGG", "TOMATO"]) > 0

        # --- 0. ENDGAME: Rush to shed and DROP everything ---
        if is_endgame and (carrying_sellable or carrying_animal):
            if (ux, uy) in SHED_ADJACENT:
                return ["DROP"], (ux, uy)
            d = _find_step((ux, uy), SHED_CENTER, board_size)
            return ([d] if d else ["PASS"]), SHED_CENTER

        # --- 1. Current Tile Interactions ---
        if isinstance(tile, dict):
            # Animal tile: harvest yield, collect fertilizer, feed and care
            if "animal" in tile:
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"], (ux, uy)
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not tile.get("cared_today", False):
                    return ["CARE"], (ux, uy)

            # Empty Pasture: place animal if holding one
            if tile.get("kind") == "PASTURE" and "animal" not in tile and carrying_animal:
                if carrying_cow:
                    return ["PLACE", "COW"], (ux, uy)
                if carrying_sheep:
                    return ["PLACE", "SHEEP"], (ux, uy)

            # Plant tile: harvest, fertilize melons, water
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

        # Empty pasture tile spot: build pasture if animal waiting
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)
        if tile is None and (ux, uy) in PASTURE_LOCATIONS and _get_quadrant(ux, uy, board_size) in unlocked:
            if animals_in_shed > 0 or carrying_animal:
                return ["BUILD_PASTURE"], (ux, uy)

        # Plant seed on empty ground
        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in PASTURE_LOCATIONS:
            if _get_quadrant(ux, uy, board_size) in unlocked:
                best_crop = self.select_crop(day, remaining_days, farm["money"])
                if best_crop and seeds.get(best_crop, 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return ["PLANT", best_crop], (ux, uy)
                for alt_crop in ["STRAWBERRY", "MELON", "CARROT", "WHEAT"]:
                    if seeds.get(alt_crop, 0) > 0:
                        claimed_tiles.add((ux, uy))
                        return ["PLANT", alt_crop], (ux, uy)

        # --- 2. Shed Pickup when adjacent ---
        if (ux, uy) in SHED_ADJACENT and not carrying_animal:
            # Priority A: Pick up animal from shed if pastures exist
            animals_on_board = sum(1 for row in farm["tiles"] for c in row if isinstance(c, dict) and "animal" in c)
            pastures_built = sum(1 for row in farm["tiles"] for c in row if isinstance(c, dict) and c.get("kind") == "PASTURE")
            if pastures_built > animals_on_board:
                if shed.get("COW", 0) > 0:
                    return ["PICKUP", "COW", 1], (ux, uy)
                if shed.get("SHEEP", 0) > 0:
                    return ["PICKUP", "SHEEP", 1], (ux, uy)
            
            # Priority B: Pick up wheat to feed hungry animals on board
            has_unfed_animals = any(
                isinstance(farm["tiles"][y][x], dict)
                and "animal" in farm["tiles"][y][x]
                and not farm["tiles"][y][x].get("fed_today", False)
                for y in range(board_size) for x in range(board_size)
            )
            if has_unfed_animals and shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
                return ["PICKUP", "WHEAT", 2], (ux, uy)
                
            # Priority C: Pick up fertilizer for melons
            if shed.get("FERTILIZER", 0) > 0 and inv.get("FERTILIZER", 0) == 0:
                has_unfert_melons = any(
                    isinstance(farm["tiles"][y][x], dict)
                    and farm["tiles"][y][x].get("kind") == "PLANT"
                    and farm["tiles"][y][x].get("crop") == "MELON"
                    and farm["tiles"][y][x].get("fertilized_until_day", -1) < day
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfert_melons:
                    return ["PICKUP", "FERTILIZER", 2], (ux, uy)

        # --- 3. Pathfinding to best target ---
        best_target = None
        best_prio = -1
        best_dist = 9999

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

                dist = _dist((ux, uy), (x, y))
                prio = -1

                if isinstance(t, dict):
                    if "animal" in t:
                        if t.get("yield_units", 0) > 0:
                            prio = 10
                        elif t.get("fertilizer_available", False):
                            prio = 9
                        elif not t.get("fed_today", False):
                            prio = 8
                        elif not t.get("cared_today", False):
                            prio = 7
                    elif t.get("kind") == "PASTURE" and "animal" not in t and carrying_animal:
                        prio = 10
                    elif t.get("kind") == "PLANT":
                        crop = t.get("crop", "")
                        age = day - t.get("planted_day", 0)
                        c_data = CROPS.get(crop, {})
                        peak = age >= c_data.get("max_yield_day", 99)
                        if (peak or is_endgame) and t.get("yield_units", 0) > 0:
                            prio = 6
                        elif not t.get("watered_today", False):
                            prio = 5
                        elif crop == "MELON" and inv.get("FERTILIZER", 0) > 0:
                            prio = 4
                    elif t.get("kind") == "WEED":
                        prio = 2
                elif t is None:
                    if (x, y) in PASTURE_LOCATIONS and (animals_in_shed > 0 or carrying_animal):
                        prio = 7
                    elif (x, y) not in PASTURE_LOCATIONS and remaining_days > 1:
                        prio = 3

                # Check if moving towards shed is useful
                if (x, y) in SHED_ADJACENT and not carrying_animal:
                    if animals_in_shed > 0:
                        prio = max(prio, 6)
                    elif shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
                        prio = max(prio, 5)

                if prio > best_prio or (prio == best_prio and dist < best_dist):
                    best_prio = prio
                    best_dist = dist
                    best_target = (x, y)

        if best_target and best_target != (ux, uy):
            step_dir = _find_step((ux, uy), best_target, board_size)
            if step_dir:
                claimed_tiles.add(best_target)
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
        hires_today = farm.get("hires_today", 0)

        market_orders = []

        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        total_sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_animals = total_cows + total_sheep
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)

        # =========================================================================
        # 1. Day 0 Master Opening (Majkel1337 / Seb Strategy)
        # =========================================================================
        if step <= 1 and len(unlocked) == 1 and money >= 2500:
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

        # =========================================================================
        # 2. Daily Labor Scaling (Hour 1 & 2):
        # Hands expire every 24 hours! Must re-hire every morning!
        # =========================================================================
        if (hour == 1 or hour == 2) and remaining_days >= 2:
            target_hires = 5
            if money >= 500:
                target_hires = 7
            if money >= 1500:
                target_hires = 9
            if money >= 5000:
                target_hires = 12
            if money >= 15000:
                target_hires = 15

            while hires_today < target_hires and len(market_orders) < 8 and money >= 50:
                market_orders.append(["HIRE"])
                hires_today += 1
                money -= 50

        # =========================================================================
        # 3. Market Sales: Sliced Sales + Fertilizer Monetization
        # =========================================================================
        is_endgame_liquidation = remaining_steps <= 18 or (day == 29 and hour >= 18)
        wheat_reserve = (total_animals + animals_in_shed) * 3

        sell_schedule = [
            ("FERTILIZER", 10),
            ("MILK", 5),
            ("WOOL", 4),
            ("STRAWBERRY", 6),
            ("MELON", 5),
            ("EGG", 10),
            ("CARROT", 15),
            ("TOMATO", 10),
            ("WHEAT", 15),
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

        # =========================================================================
        # 4. Strictly Gated Land Expansion:
        # NEVER before Day 6! Protect cash for daily labor and livestock.
        # =========================================================================
        if "NE" not in unlocked and money >= 1200 and 6 <= day <= 12 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and 10 <= day <= 18 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif "SE" not in unlocked and money >= 4000 and 15 <= day <= 24 and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 3000

        # =========================================================================
        # 5. Animal Purchases:
        # Target: 4 animals early -> 8 animals Day 6+ -> 14 animals Day 10+
        # =========================================================================
        target_animals = 4
        if day >= 6 and len(unlocked) >= 2:
            target_animals = 8
        if day >= 10 and len(unlocked) >= 3:
            target_animals = 14
        if day >= 16 and len(unlocked) >= 4:
            target_animals = 18

        if (total_animals + animals_in_shed) < target_animals and day <= 18 and remaining_days >= 7 and len(market_orders) < 9:
            if total_cows <= total_sheep and money >= 600:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400
            elif money >= 700:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500

        # =========================================================================
        # 6. Wheat Feed Procurement:
        # Guarantee enough wheat to feed all animals daily
        # =========================================================================
        needed_feed = (total_animals + animals_in_shed) * 3
        current_wheat = shed.get("WHEAT", 0)
        if (total_animals + animals_in_shed) > 0 and current_wheat < needed_feed and money >= 100 and len(market_orders) < 9:
            buy_wheat_amt = min(12, needed_feed - current_wheat + 4)
            if buy_wheat_amt > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_amt])
                money -= buy_wheat_amt * 25

        # =========================================================================
        # 7. Seed Procurement:
        # Strawberry focus starting Day 6!
        # =========================================================================
        chosen_crop = self.select_crop(day, remaining_days, money)
        if chosen_crop and remaining_days >= 2 and len(market_orders) < 9:
            current_count = seeds.get(chosen_crop, 0)
            if current_count < 15:
                seed_cost = CROPS[chosen_crop]["seed"]
                buy_n = min(8, max(0, int(money // seed_cost)))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        # =========================================================================
        # 8. Units Action Execution: Farmer + All Hands
        # =========================================================================
        claimed_tiles = set()
        farmer_act, _ = self.plan_unit(0, farm["farmer"], farm, private, day, hour, step, claimed_tiles)

        hands_acts = []
        for h_idx, h_pos in enumerate(farm.get("hands", [])):
            h_act, _ = self.plan_unit(h_idx + 1, h_pos, farm, private, day, hour, step, claimed_tiles)
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }


_champion_agent = ChampionAgentV5()


def agent(obs):
    \"\"\"Kaggle Environments Entry Point.\"\"\"
    return _champion_agent.act(obs)
"""

with open("06_kaggriculture/main.py", "w", encoding="utf-8") as f:
    f.write(code)
print("Successfully wrote Champion Agent v5 into 06_kaggriculture/main.py!")
