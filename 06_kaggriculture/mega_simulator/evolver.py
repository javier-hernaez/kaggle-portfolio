import json
import os
import sys
import time
import random
from typing import List, Dict, Tuple

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from mega_simulator.agent_genome import Genome, mutate, crossover
from mega_simulator.arena import Arena

HOF_FILE = "06_kaggriculture/mega_simulator/hall_of_fame.json"
MAIN_PY_FILE = "06_kaggriculture/main.py"


def save_hall_of_fame(history: List[Dict]):
    with open(HOF_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_hall_of_fame() -> List[Dict]:
    if os.path.exists(HOF_FILE):
        try:
            with open(HOF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def export_genome_to_main_py(g: Genome):
    """Exports the current best evolved genome to main.py."""
    code = f'''"""
Deep Grandmaster Agent (Industrial +1,000 Self-Play Evolved).
Chromosome Generation: {g.name}

Evolved Parameters:
- NE Land: Day {g.land_ne_min_day}+ (Min Money: ${g.land_ne_min_money})
- SW Land: Day {g.land_sw_min_day}+ (Min Money: ${g.land_sw_min_money})
- Buy SE Land: {g.buy_se_land} (Day {g.land_se_min_day}, Min ${g.land_se_min_money})
- Animals: {g.target_cows} Cows, {g.target_sheep} Sheep (Cutoff Day {g.animal_end_day})
- Strawberries: Base {g.target_strawberries} (Boost: +{g.shop_icecream_straw_boost} if IceCream/Smoothie shop opens)
- Sheep Synergy: +{g.shop_yarn_sheep_boost} if Yarn Store opens
- Labor: Early {g.labor_early}, Mid {g.labor_mid} (Day {g.labor_mid_day}), Late {g.labor_late} (Day {g.labor_late_day})
- Price Arbitrage: Throttle < {g.price_throttle_ratio*100:.0f}%, Burst >= {g.price_burst_ratio*100:.0f}%
- Feed Reserve Mult: {g.wheat_reserve_mult}x
"""

CROPS = {{
    "WHEAT":      {{"seed": 10,  "first_yield_day": 2,  "max_yield_day": 4,  "interval": 0, "ongoing": False}},
    "CARROT":     {{"seed": 20,  "first_yield_day": 2,  "max_yield_day": 3,  "interval": 0, "ongoing": False}},
    "TOMATO":     {{"seed": 50,  "first_yield_day": 8,  "max_yield_day": 8,  "interval": 1, "ongoing": True}},
    "STRAWBERRY": {{"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "ongoing": True}},
    "MELON":      {{"seed": 80,  "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "ongoing": False}},
}}

BASE_PRICES = {{
    "WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
    "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100
}}

ANIMALS = {{
    "COW":   {{"cost": 400, "product": "MILK"}},
    "SHEEP": {{"cost": 500, "product": "WOOL"}},
}}

MOVES = {{
    (0, -1): "NORTH",
    (0,  1): "SOUTH",
    (1,  0): "EAST",
    (-1, 0): "WEST",
}}

PASTURE_LOCATIONS = [
    (4, 4), (3, 4), (4, 3), (3, 3),
    (5, 4), (5, 3), (4, 2), (5, 2),
    (6, 4), (6, 3), (7, 4),
    (2, 4), (3, 5), (4, 5)
]

SHED_ADJACENT = {{(4, 4), (5, 4), (4, 5), (5, 5)}}
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
        if day < {g.strawberry_start_day}:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 2: Strawberry Engine
        if {g.strawberry_start_day} <= day <= {g.strawberry_end_day}:
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
        inv = private["inventories"][unit_idx] if unit_idx < len(private.get("inventories", [])) else {{}}
        shed = private.get("shed", {{}})
        seeds = private.get("seeds", {{}})
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
                c_data = CROPS.get(crop, {{}})
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
                        c_data = CROPS.get(t["crop"], {{}})
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
        shed = private.get("shed", {{}})
        seeds = private.get("seeds", {{}})
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        market_prices = obs.get("market", {{}}).get("prices", {{}})
        unlocked_shops = obs.get("town", {{}}).get("unlocked_shops", [])

        # Dynamic Town Shop Adaptation
        has_ice_cream_or_smoothie = any(s in ["ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "BRUNCH_SPOT"] for s in unlocked_shops)
        has_yarn_store = "YARN_STORE" in unlocked_shops

        eff_target_strawberries = {g.target_strawberries} + ({g.shop_icecream_straw_boost} if has_ice_cream_or_smoothie else 0)
        eff_target_sheep = {g.target_sheep} + ({g.shop_yarn_sheep_boost} if has_yarn_store else 0)

        market_orders = []

        total_cows = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        total_sheep = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        total_animals = total_cows + total_sheep
        animals_in_shed = shed.get("COW", 0) + shed.get("SHEEP", 0)
        strawberry_count = sum(1 for row in farm["tiles"] for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")

        # 1. Day 0 Master Opening
        if step == 0 or (step == 1 and len(unlocked) == 1 and money >= 2500):
            return {{
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
            }}

        # 2. Price-Aware Sales
        is_endgame_liquidation = (day >= 29 and hour >= 16) or (remaining_steps <= 12)
        wheat_reserve = (total_animals + animals_in_shed) * {g.wheat_reserve_mult}

        sell_schedule = [
            ("FERTILIZER", {g.batch_fertilizer}),
            ("MILK", {g.batch_milk}),
            ("WOOL", {g.batch_wool}),
            ("STRAWBERRY", {g.batch_strawberry}),
            ("MELON", {g.batch_melon}),
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

            if not is_endgame_liquidation and curr_price < base_p * {g.price_throttle_ratio}:
                effective_slice = max(1, base_slice // 2)
            elif curr_price >= base_p * {g.price_burst_ratio}:
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
            target_hires = {g.labor_early}
            if day >= {g.labor_mid_day}:
                target_hires = {g.labor_mid}
            if day >= {g.labor_late_day}:
                target_hires = {g.labor_late}

            while hires_today < target_hires and len(market_orders) < 9 and money >= 20:
                market_orders.append(["HIRE"])
                hires_today += 1
                money -= 20

        # 4. Gated Land Expansion
        if "NE" not in unlocked and money >= {g.land_ne_min_money} and day >= {g.land_ne_min_day} and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= {g.land_sw_min_money} and day >= {g.land_sw_min_day} and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif {g.buy_se_land} and "SE" not in unlocked and money >= {g.land_se_min_money} and day >= {g.land_se_min_day} and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 3000

        # 5. Balanced Animal Purchases
        total_capacity = len(PASTURE_LOCATIONS)
        if (total_animals + animals_in_shed) < total_capacity and day <= {g.animal_end_day} and len(market_orders) < 9:
            if total_sheep < eff_target_sheep and (total_cows >= total_sheep * {g.cow_to_sheep_ratio} or total_cows >= {g.target_cows}) and money >= 600:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            elif total_cows < {g.target_cows} and money >= 500:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400

        # 6. Wheat Feed Procurement
        needed_feed = (total_animals + animals_in_shed) * {g.wheat_reserve_mult}
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
                max_batch = {g.strawberry_seed_batch} if chosen_crop == "STRAWBERRY" else 6
                buy_n = min(max_batch, int(money // seed_cost))
                if buy_n > 0:
                    market_orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        # 8. Squad Specialization Execution
        claimed_tiles = set()
        shared_state = {{
            "animal_pickup_claimed": False,
            "strawberries": strawberry_count
        }}

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

        return {{
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }}


_champion_brain = DeepChampionAgent()


def agent(obs):
    """Kaggle Environments Entry Point."""
    return _champion_brain.act(obs)
'''
    with open(MAIN_PY_FILE, "w", encoding="utf-8") as f:
        f.write(code)
    print(f"[Export] Crowned and exported new champion '{g.name}' to {MAIN_PY_FILE}!")


class MegaEvolver:
    """Industrial Evolutionary System for +1,000 Matches."""
    def __init__(self, workers: int = 6):
        self.arena = Arena(max_workers=workers)
        self.hof = load_hall_of_fame()

    def run_mega_evolution(self, total_target_games: int = 1000, population_size: int = 10, seeds_per_duel: int = 5):
        """
        Runs massive generation-based evolutionary search reaching total_target_games.
        Each mirror duel with seeds_per_duel runs 2 * seeds_per_duel games (e.g. 10 games).
        """
        games_per_candidate = seeds_per_duel * 2
        candidates_per_gen = population_size - 2
        games_per_gen = candidates_per_gen * games_per_candidate
        total_generations = max(5, int(total_target_games // games_per_gen))

        base_seeds = [42, 1039, 2036, 3033, 4030, 5027, 6024, 7021, 8018, 9015, 1123, 2345, 3456, 4567, 5678, 6789]

        # Seed population
        current_champ = Genome(name="SuperGrandmaster_Base")
        if self.hof:
            current_champ = Genome.from_dict(self.hof[-1]["genome"])
            print(f"[MegaEvolver] Loaded reigning champion '{current_champ.name}' from Hall of Fame.")
        else:
            print(f"[MegaEvolver] Initializing from base 28-gene genome.")

        population = [current_champ]
        for i in range(1, population_size):
            population.append(mutate(current_champ, mutation_rate=0.35))

        print("=" * 70)
        print("  INICIANDO MEGA-SIMULADOR INDUSTRIAL (+1,000 PARTIDAS DE AUTO-JUEGO)")
        print(f"  Objetivo de Partidas: ~{total_generations * games_per_gen:,} ({total_generations} generaciones x {games_per_gen} partidas)")
        print(f"  Tamaño de Población: {population_size} | Semillas por Duelo: {seeds_per_duel} (Mirror Match x2)")
        print(f"  Hilos de Ejecución Paralela: {self.arena.max_workers}")
        print("=" * 70)

        total_simulated = 0
        all_time_high_score = 0.0

        for gen in range(1, total_generations + 1):
            t_gen_start = time.perf_counter()
            round_seeds = base_seeds[(gen % 5): (gen % 5) + seeds_per_duel]

            print(f"\n--- [Generación {gen:02d}/{total_generations:02d}] Evaluando Población ({len(population)} variantes) ---")

            # Champion is population[0]
            champ = population[0]
            ranked_candidates = []

            for idx, candidate in enumerate(population[1:], start=1):
                res = self.arena.duel(candidate, champ, round_seeds)
                total_simulated += res["total_games"]

                fitness = res["avg_score_a"] + 10000.0 * (res["winrate_a"] - 0.5)
                ranked_candidates.append({
                    "genome": candidate,
                    "res": res,
                    "fitness": fitness,
                    "avg_score": res["avg_score_a"],
                    "winrate": res["winrate_a"],
                    "margin": res["net_margin"]
                })

                prefix = "[HOT]" if res["is_a_superior"] else "     "
                print(f"  {prefix} Var {idx:02d} ({candidate.name[:14]:14s}): Winrate {res['winrate_a']*100:5.1f}% | Media ${res['avg_score_a']:6,.0f} vs ${res['avg_score_b']:6,.0f} (Margen: ${res['net_margin']:+6,.0f})")

            # Sort by fitness
            ranked_candidates.sort(key=lambda x: x["fitness"], reverse=True)
            top_candidate = ranked_candidates[0]

            # Check if top candidate defeats champion in confirmation
            if top_candidate["res"]["is_a_superior"]:
                print(f"\n  [CHECK] Verificación Profunda del Líder: {top_candidate['genome'].name} vs {champ.name} (8 semillas = 16 partidas)...")
                verify_seeds = base_seeds[:8]
                v_res = self.arena.duel(top_candidate["genome"], champ, verify_seeds)
                total_simulated += v_res["total_games"]

                print(f"  -> Resultado: Winrate {v_res['winrate_a']*100:.1f}% | Margen: ${v_res['net_margin']:+,.0f} | Media: ${v_res['avg_score_a']:,.0f}")

                if v_res["is_a_superior"]:
                    print(f"  [NEW CHAMPION] ¡NUEVO CAMPEÓN SUPREMO: {top_candidate['genome'].name}!")
                    champ = top_candidate["genome"]
                    if v_res["avg_score_a"] > all_time_high_score:
                        all_time_high_score = v_res["avg_score_a"]

                    # Save record and export
                    rec = {
                        "timestamp": time.time(),
                        "generation": gen,
                        "genome": champ.to_dict(),
                        "avg_score": v_res["avg_score_a"],
                        "net_margin": v_res["net_margin"],
                        "winrate": v_res["winrate_a"],
                        "total_games_so_far": total_simulated,
                    }
                    self.hof.append(rec)
                    save_hall_of_fame(self.hof)
                    export_genome_to_main_py(champ)
                else:
                    print(f"  El candidato no superó la confirmación. El campeón {champ.name} se mantiene.")
            else:
                print(f"  El campeón {champ.name} retiene el liderazgo en la Generación {gen}.")

            # Next generation breeding:
            # 1. Elitism: Champ + best candidate survive untouched
            new_pop = [champ, top_candidate["genome"]]

            # 2. Crossover & Mutate remaining slots
            fit_parents = [champ, top_candidate["genome"]] + [c["genome"] for c in ranked_candidates[:3]]
            while len(new_pop) < population_size:
                p1, p2 = random.sample(fit_parents, 2)
                child = crossover(p1, p2)
                if random.random() < 0.6:
                    child = mutate(child, mutation_rate=0.25)
                child.name = f"Gen{gen+1}_Ind{len(new_pop)}"
                new_pop.append(child)

            population = new_pop
            t_gen_elapsed = time.perf_counter() - t_gen_start
            print(f"  [Gen {gen:02d} completada en {t_gen_elapsed:.1f}s | Partidas acumuladas: {total_simulated:,}]")

        print("\n" + "=" * 70)
        print(f"  FIN DEL MEGA-SIMULADOR INDUSTRIAL.")
        print(f"  Total Partidas Simuladas: {total_simulated:,}")
        print(f"  Campeón Supremo Actual: '{champ.name}'")
        print("=" * 70)
        return champ
