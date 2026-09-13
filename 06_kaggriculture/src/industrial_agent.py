"""
Industrial Strategic Agent v2: Farm & Livestock Enterprise (Optimized).
Features:
- Turn 1 Opening Gambit: NE land expansion + 5 Hires + Cow Pastures + Melons/Strawberries/Wheat.
- Expanded Livestock Capacity: Up to 10 automated pastures with daily wheat feeding, CARE, and milk collection.
- Fertilizer Supply Chain: Workers pick up fertilizer from shed/cows and fertilize Melons to double yield.
- Sliced Market Selling: Sells in batches of 3-5 to maintain elevated prices above equilibrium.
- Scaled Labor: Dynamically hires up to 12 farm hands as daily cash flow expands.
- Full Endgame Liquidation: Complete clearance of all crops and shed items by Day 29.
"""

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
}

MOVES = {
    (0, -1): "NORTH",
    (0, 1): "SOUTH",
    (1, 0): "EAST",
    (-1, 0): "WEST",
}

PASTURE_LOCATIONS = [
    (3, 4), (4, 3), (3, 3), (3, 5), (5, 3),
    (2, 3), (2, 4), (2, 5), (3, 2), (4, 2)
]


def _get_quadrant(x, y, board_size=10):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _is_shed_adj(x, y):
    return (x, y) in {(4, 4), (5, 4), (4, 5), (5, 5)}


def _find_step(start_pos, target_pos, board_size=10):
    sx, sy = start_pos
    tx, ty = target_pos
    if (sx, sy) == (tx, ty):
        return None
    dx = tx - sx
    dy = ty - sy
    if abs(dx) >= abs(dy) and dx != 0:
        step_x = 1 if dx > 0 else -1
        nx, ny = sx + step_x, sy
        if 0 <= nx < board_size:
            return MOVES.get((step_x, 0))
    if dy != 0:
        step_y = 1 if dy > 0 else -1
        nx, ny = sx, sy + step_y
        if 0 <= ny < board_size:
            return MOVES.get((0, step_y))
    if dx != 0:
        step_x = 1 if dx > 0 else -1
        return MOVES.get((step_x, 0))
    return None


class IndustrialAgent:
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
        if day <= 17:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["STRAWBERRY"]["seed"]:
                return "STRAWBERRY"
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
        elif day <= 24:
            if money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
        else:
            if remaining_days >= 3 and money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if remaining_days >= 2 and money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
        return None

    def plan_unit_turn(self, unit_idx, unit_pos, farm, private, day, hour, step, claimed_tiles):
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        tile = farm["tiles"][uy][ux]
        inv = private["inventories"][unit_idx] if unit_idx < len(private["inventories"]) else {}
        shed = private.get("shed", {})
        seeds = private.get("seeds", {})
        remaining_steps = 720 - step
        remaining_days = 30 - day

        # 1. Tile actions at current position
        # A) Animal interactions
        if isinstance(tile, dict) and "animal" in tile:
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"], (ux, uy)
            if tile.get("fertilizer_available", False):
                return ["COLLECT_FERTILIZER"], (ux, uy)
            if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                return ["FEED"], (ux, uy)
            if not tile.get("cared_today", False):
                return ["CARE"], (ux, uy)

        # B) Empty pasture: Place animal if holding COW/SHEEP
        if isinstance(tile, dict) and tile.get("kind") == "PASTURE" and "animal" not in tile:
            if inv.get("COW", 0) > 0:
                return ["PLACE", "COW"], (ux, uy)
            if inv.get("SHEEP", 0) > 0:
                return ["PLACE", "SHEEP"], (ux, uy)

        # C) Designated Pasture location needing BUILD_PASTURE
        if (ux, uy) in PASTURE_LOCATIONS and tile is None:
            total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
            if total_cows < len(PASTURE_LOCATIONS) and (shed.get("COW", 0) > 0 or inv.get("COW", 0) > 0 or farm["money"] >= 400):
                return ["BUILD_PASTURE"], (ux, uy)

        # D) Plant interactions
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile["crop"]
            c_data = CROPS[crop]
            age = day - tile["planted_day"]
            yield_units = tile.get("yield_units", 0)

            is_peak = age >= c_data["max_yield_day"]
            is_endgame = remaining_steps <= 36
            if (is_peak or is_endgame) and yield_units > 0:
                if not c_data["ongoing"] or age >= c_data["first_yield_day"]:
                    return ["HARVEST"], (ux, uy)

            # Fertilize melon if holding fertilizer
            if crop == "MELON" and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day:
                return ["FERTILIZER"], (ux, uy)

            if not tile.get("watered_today", False):
                return ["WATER"], (ux, uy)

        # E) Plant seed on empty tile (excluding reserved pasture tiles)
        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in PASTURE_LOCATIONS:
            best_crop = self.select_crop(day, remaining_days, farm["money"])
            if best_crop and seeds.get(best_crop, 0) > 0:
                claimed_tiles.add((ux, uy))
                return ["PLANT", best_crop], (ux, uy)
            for s_name in ["MELON", "STRAWBERRY", "CARROT", "WHEAT"]:
                if seeds.get(s_name, 0) > 0:
                    claimed_tiles.add((ux, uy))
                    return ["PLANT", s_name], (ux, uy)

        # F) Weeds
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ["DIG"], (ux, uy)

        # G) Shed pickup when adjacent:
        if _is_shed_adj(ux, uy):
            # Pickup cow if empty pasture exists
            has_empty_pasture = any(farm["tiles"][py][px] == {"kind": "PASTURE"} for px, py in PASTURE_LOCATIONS if farm["tiles"][py][px] is not None)
            if has_empty_pasture and shed.get("COW", 0) > 0 and inv.get("COW", 0) == 0:
                return ["PICKUP", "COW", 1], (ux, uy)
            if shed.get("WHEAT", 0) > 0 and inv.get("WHEAT", 0) == 0:
                return ["PICKUP", "WHEAT", 2], (ux, uy)
            if shed.get("FERTILIZER", 0) > 0 and inv.get("FERTILIZER", 0) == 0:
                return ["PICKUP", "FERTILIZER", 2], (ux, uy)

        # 2. Pathfinding to next target
        best_target = None
        best_dist = 9999
        best_prio = -1

        unlocked = set(farm.get("unlocked_quadrants", ["NW"]))

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

                dist = abs(x - ux) + abs(y - uy)
                prio = -1

                if isinstance(t, dict) and "animal" in t:
                    if t.get("yield_units", 0) > 0:
                        prio = 6
                    elif not t.get("fed_today", False):
                        prio = 5
                    elif t.get("fertilizer_available", False):
                        prio = 4
                elif isinstance(t, dict) and t.get("kind") == "PLANT":
                    c_data = CROPS[t["crop"]]
                    age = day - t["planted_day"]
                    if (age >= c_data["max_yield_day"] or remaining_steps <= 36) and t.get("yield_units", 0) > 0:
                        prio = 5
                    elif not t.get("watered_today", False):
                        prio = 4
                elif (x, y) in PASTURE_LOCATIONS and (t is None or (isinstance(t, dict) and "animal" not in t)):
                    prio = 3
                elif t is None and remaining_days > 2 and (x, y) not in PASTURE_LOCATIONS:
                    prio = 2
                elif isinstance(t, dict) and t.get("kind") == "WEED":
                    prio = 1

                if prio > best_prio or (prio == best_prio and dist < best_dist):
                    best_prio = prio
                    best_dist = dist
                    best_target = (x, y)

        if best_target is not None and best_target != (ux, uy):
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

        market_orders = []

        # 1. Opening Gambit
        if step == 0 or (step == 1 and len(unlocked) == 1 and money >= 2000):
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["BUY_LAND"])
            market_orders.append(["BUY_ANIMAL", "COW", 2])
            market_orders.append(["BUY_SEED", "WHEAT", 6])
            market_orders.append(["BUY_SEED", "MELON", 4])
            market_orders.append(["BUY_SEED", "STRAWBERRY", 3])
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": market_orders,
            }

        # 2. Market Operations: Sliced Sales
        is_endgame_liquidation = (day >= 29 and hour >= 16) or (remaining_steps <= 12)

        for prod, slice_size in [
            ("MELON", 5),
            ("MILK", 5),
            ("WOOL", 4),
            ("STRAWBERRY", 6),
            ("FERTILIZER", 5),
            ("WHEAT", 15),
            ("CARROT", 15),
            ("TOMATO", 10),
            ("EGG", 10),
        ]:
            qty = shed.get(prod, 0)
            if qty > 0 and len(market_orders) < 8:
                sell_qty = qty if is_endgame_liquidation else min(qty, slice_size)
                market_orders.append(["SELL", prod, sell_qty])

        # 3. Land Expansion:
        if "NE" not in unlocked and money >= 1100 and day <= 15:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and day <= 18:
            market_orders.append(["BUY_LAND"])
            money -= 2000

        # 4. Daily Hires at hour 1 (scaled with bankroll):
        hires_today = farm.get("hires_today", 0)
        if (hour == 1 or hour == 2) and remaining_days > 2:
            target_hires = 5
            if money >= 2000:
                target_hires = 7
            if money >= 6000:
                target_hires = 10
            if money >= 15000:
                target_hires = 12

            while hires_today < target_hires and len(market_orders) < 10:
                market_orders.append(["HIRE"])
                hires_today += 1

        # 5. Animal Purchases:
        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        cows_pending = shed.get("COW", 0)
        if (total_cows + cows_pending) < len(PASTURE_LOCATIONS) and money >= 600 and day <= 16 and len(market_orders) < 10:
            market_orders.append(["BUY_ANIMAL", "COW", 1])
            money -= 400

        # 6. Wheat for animal feed:
        if (total_cows + cows_pending) > 0 and shed.get("WHEAT", 0) < 8 and money >= 150 and len(market_orders) < 10:
            market_orders.append(["BUY_PRODUCT", "WHEAT", 4])
            money -= 100

        # 7. Seed Purchases:
        chosen_crop = self.select_crop(day, remaining_days, money)
        if chosen_crop and remaining_days > 2 and len(market_orders) < 10:
            current_count = seeds.get(chosen_crop, 0)
            if current_count < 10:
                seed_cost = CROPS[chosen_crop]["seed"]
                buy_n = min(6, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        # 8. Units turns
        claimed_tiles = set()
        farmer_act, _ = self.plan_unit_turn(0, farm["farmer"], farm, private, day, hour, step, claimed_tiles)

        hands_acts = []
        for h_idx, h_pos in enumerate(farm.get("hands", [])):
            h_act, _ = self.plan_unit_turn(h_idx + 1, h_pos, farm, private, day, hour, step, claimed_tiles)
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }


_industrial_agent = IndustrialAgent()


def industrial_agent(obs):
    return _industrial_agent.act(obs)
