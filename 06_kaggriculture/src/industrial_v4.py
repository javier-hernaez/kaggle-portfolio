"""
Industrial Strategic Agent v4: Full Empire & Precision Logistics.
Optimizations over v3:
- 100% Board Coverage (SE Quadrant Expansion): Expands to all 4 quadrants (100 tiles total).
- Endgame Shed Delivery (Field-to-Shed Clearance): Units carrying inventory in Day 29 march to the shed and DROP to monetize all field assets before Step 720.
- Fertilizer Price Optimization: Halts selling fertilizer when price drops below $35, channeling it to plants instead.
- Geese Coops (Egg Production): Adds 2 Geese to capture daily egg demand from Bakeries & Brunch Spots.
- Expanded Pastures (up to 14 animals): Cow & Sheep capacity expanded into unlocked territory.
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
    (2, 3), (2, 4), (2, 5), (3, 2), (4, 2),
    (6, 3), (6, 4), (3, 6), (4, 6)
]

COOP_LOCATIONS = [(1, 3), (1, 4)]


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


class EmpireAgent:
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
        is_endgame_clearance = (day >= 29 and hour >= 16) or (remaining_steps <= 14)

        # 0. Endgame field-to-shed clearance:
        # If carrying high value produce at endgame, head to shed and DROP!
        has_sellable_inv = sum(inv.get(k, 0) for k in ["MILK", "WOOL", "STRAWBERRY", "MELON", "CARROT", "FERTILIZER", "EGG", "WHEAT"]) > 0
        if is_endgame_clearance and has_sellable_inv:
            if _is_shed_adj(ux, uy):
                return ["DROP"], (ux, uy)
            step_to_shed = _find_step((ux, uy), (4, 4), board_size)
            if step_to_shed:
                return [step_to_shed], (4, 4)

        # 1. Tile actions at current position
        # A) Animal interactions (Pasture / Coop)
        if isinstance(tile, dict) and "animal" in tile:
            if tile.get("yield_units", 0) > 0:
                return ["HARVEST"], (ux, uy)
            if tile.get("fertilizer_available", False):
                return ["COLLECT_FERTILIZER"], (ux, uy)
            if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                return ["FEED"], (ux, uy)
            if not tile.get("cared_today", False):
                return ["CARE"], (ux, uy)

        # B) Empty structure: Place animal
        if isinstance(tile, dict) and "animal" not in tile:
            kind = tile.get("kind")
            if kind == "PASTURE":
                if inv.get("COW", 0) > 0:
                    return ["PLACE", "COW"], (ux, uy)
                if inv.get("SHEEP", 0) > 0:
                    return ["PLACE", "SHEEP"], (ux, uy)
            elif kind == "COOP":
                if inv.get("GOOSE", 0) > 0:
                    return ["PLACE", "GOOSE"], (ux, uy)

        # C) Build Pasture / Coop
        if (ux, uy) in PASTURE_LOCATIONS and tile is None:
            total_animals = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and "animal" in t)
            animals_ready = shed.get("COW", 0) + shed.get("SHEEP", 0) + inv.get("COW", 0) + inv.get("SHEEP", 0)
            if total_animals < len(PASTURE_LOCATIONS) and (animals_ready > 0 or farm["money"] >= 400):
                return ["BUILD_PASTURE"], (ux, uy)

        if (ux, uy) in COOP_LOCATIONS and tile is None:
            if shed.get("GOOSE", 0) > 0 or inv.get("GOOSE", 0) > 0 or farm["money"] >= 300:
                return ["BUILD_COOP"], (ux, uy)

        # D) Plant interactions
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop = tile["crop"]
            c_data = CROPS[crop]
            age = day - tile["planted_day"]
            yield_units = tile.get("yield_units", 0)

            is_peak = age >= c_data["max_yield_day"]
            if (is_peak or is_endgame_clearance) and yield_units > 0:
                if not c_data["ongoing"] or age >= c_data["first_yield_day"]:
                    return ["HARVEST"], (ux, uy)

            if crop in ("MELON", "STRAWBERRY") and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < day:
                return ["FERTILIZER"], (ux, uy)

            if not tile.get("watered_today", False):
                return ["WATER"], (ux, uy)

        # E) Plant seed on empty tile
        reserved = set(PASTURE_LOCATIONS) | set(COOP_LOCATIONS)
        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in reserved:
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
            has_empty_pasture = any(
                farm["tiles"][py][px] == {"kind": "PASTURE"}
                for px, py in PASTURE_LOCATIONS
                if 0 <= px < board_size and 0 <= py < board_size and farm["tiles"][py][px] is not None
            )
            if has_empty_pasture and inv.get("COW", 0) == 0 and inv.get("SHEEP", 0) == 0:
                if shed.get("COW", 0) > 0:
                    return ["PICKUP", "COW", 1], (ux, uy)
                if shed.get("SHEEP", 0) > 0:
                    return ["PICKUP", "SHEEP", 1], (ux, uy)

            has_empty_coop = any(
                farm["tiles"][py][px] == {"kind": "COOP"}
                for px, py in COOP_LOCATIONS
                if 0 <= px < board_size and 0 <= py < board_size and farm["tiles"][py][px] is not None
            )
            if has_empty_coop and inv.get("GOOSE", 0) == 0 and shed.get("GOOSE", 0) > 0:
                return ["PICKUP", "GOOSE", 1], (ux, uy)

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
                    if (age >= c_data["max_yield_day"] or is_endgame_clearance) and t.get("yield_units", 0) > 0:
                        prio = 5
                    elif not t.get("watered_today", False):
                        prio = 4
                elif (x, y) in reserved and (t is None or (isinstance(t, dict) and "animal" not in t)):
                    prio = 3
                elif t is None and remaining_days > 2 and (x, y) not in reserved:
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
        prices = obs.get("market", {}).get("prices", {})

        market_orders = []

        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        total_sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_geese = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "GOOSE")
        total_animals = total_cows + total_sheep + total_geese
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0) + shed.get("GOOSE", 0)

        # 1. Opening Gambit
        if step == 0 or (step == 1 and len(unlocked) == 1 and money >= 2000):
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["HIRE"])
            market_orders.append(["BUY_LAND"])
            market_orders.append(["BUY_ANIMAL", "COW", 2])
            market_orders.append(["BUY_PRODUCT", "WHEAT", 4])
            market_orders.append(["BUY_SEED", "MELON", 4])
            market_orders.append(["BUY_SEED", "STRAWBERRY", 3])
            return {
                "farmer": ["PASS"],
                "hands": [],
                "market": market_orders,
            }

        # 2. Market Operations: Sliced Sales (Anti-Crash & Price Aware)
        is_endgame_liquidation = (day >= 29 and hour >= 16) or (remaining_steps <= 12)
        wheat_reserve = (total_animals + animals_in_shed) * 2

        for prod, slice_size, min_price in [
            ("MELON", 5, 50),
            ("MILK", 5, 50),
            ("WOOL", 4, 60),
            ("STRAWBERRY", 6, 40),
            ("FERTILIZER", 5, 35),
            ("WHEAT", 15, 10),
            ("CARROT", 15, 15),
            ("EGG", 10, 20),
            ("TOMATO", 10, 20),
        ]:
            qty = shed.get(prod, 0)
            if qty <= 0 or len(market_orders) >= 8:
                continue

            current_p = prices.get(prod, 100)
            # Throttle if price is crashed unless it's endgame
            if not is_endgame_liquidation and current_p < min_price:
                continue

            if prod == "WHEAT" and not is_endgame_liquidation:
                sellable_wheat = max(0, qty - wheat_reserve)
                if sellable_wheat > 0:
                    market_orders.append(["SELL", "WHEAT", min(sellable_wheat, slice_size)])
                continue

            sell_qty = qty if is_endgame_liquidation else min(qty, slice_size)
            market_orders.append(["SELL", prod, sell_qty])

        # 3. Full 4-Quadrant Expansion (NE, SW, SE)
        if "NE" not in unlocked and money >= 1100 and day <= 15:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= 2500 and day <= 18:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif "SE" not in unlocked and money >= 5000 and day <= 20:
            market_orders.append(["BUY_LAND"])
            money -= 4000

        # 4. Daily Labor Scaling (Hour 1 & 2):
        hires_today = farm.get("hires_today", 0)
        if (hour == 1 or hour == 2) and remaining_days > 2:
            target_hires = 5
            if money >= 2000:
                target_hires = 7
            if money >= 6000:
                target_hires = 9
            if money >= 15000:
                target_hires = 12

            while hires_today < target_hires and len(market_orders) < 10:
                market_orders.append(["HIRE"])
                hires_today += 1

        # 5. Animal Purchases:
        total_capacity = len(PASTURE_LOCATIONS)
        if (total_animals + animals_in_shed) < total_capacity and day <= 16 and len(market_orders) < 10:
            if total_cows < 5 and money >= 600:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400
            elif total_sheep < 4 and money >= 700:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            elif total_geese < 2 and money >= 500:
                market_orders.append(["BUY_ANIMAL", "GOOSE", 1])
                money -= 300

        # 6. Wheat Feed Procurement:
        needed_feed = (total_animals + animals_in_shed) * 2
        current_wheat = shed.get("WHEAT", 0)
        if (total_animals + animals_in_shed) > 0 and current_wheat < needed_feed and money >= 150 and len(market_orders) < 10:
            buy_wheat_amt = min(12, needed_feed - current_wheat + 4)
            if buy_wheat_amt > 0:
                market_orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_amt])
                money -= buy_wheat_amt * 25

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

        # 8. Units Turns Execution:
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


_empire_agent = EmpireAgent()


def empire_agent(obs):
    return _empire_agent.act(obs)
