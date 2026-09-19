import copy
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

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

PASTURE_LOCATIONS = [
    (4, 4), (3, 4), (4, 3), (3, 3),
    (5, 4), (5, 3), (4, 2), (5, 2),
    (6, 4), (6, 3), (7, 4),
    (2, 4), (3, 5), (4, 5)
]

SHED_ADJACENT = {(4, 4), (5, 4), (4, 5), (5, 5)}
SHED_CENTER = (4, 4)

MOVES = {
    (0, -1): "NORTH",
    (0,  1): "SOUTH",
    (1,  0): "EAST",
    (-1, 0): "WEST",
}


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


@dataclass
class Genome:
    """The 28-gene strategic chromosome for the Grandmaster Agent."""
    name: str = "Grandmaster_Base"
    # 1-6: Land expansion
    land_ne_min_day: int = 5
    land_ne_min_money: int = 1300
    land_sw_min_day: int = 10
    land_sw_min_money: int = 2500
    buy_se_land: bool = False
    land_se_min_day: int = 16
    land_se_min_money: int = 4500

    # 7-11: Livestock herd
    target_cows: int = 10
    target_sheep: int = 5
    animal_end_day: int = 12
    cow_to_sheep_ratio: float = 2.0
    wheat_reserve_mult: int = 1

    # 12-15: Strawberry Engine
    strawberry_start_day: int = 5
    strawberry_end_day: int = 13
    target_strawberries: int = 45
    strawberry_seed_batch: int = 10

    # 16-20: Labor scaling
    labor_early: int = 7
    labor_mid: int = 8
    labor_mid_day: int = 6
    labor_late: int = 12
    labor_late_day: int = 10

    # 21-25: Market Arbitrage & Batch Slices
    price_throttle_ratio: float = 0.65
    price_burst_ratio: float = 1.15
    batch_milk: int = 6
    batch_wool: int = 4
    batch_strawberry: int = 8
    batch_fertilizer: int = 12
    batch_melon: int = 6

    # 26-28: Town Shop Synergy & Weather
    shop_icecream_straw_boost: int = 10  # boost strawberry target if icecream/smoothie shop open
    shop_yarn_sheep_boost: int = 2       # boost sheep target if yarn store open
    skip_watering_if_rained: bool = True # rain awareness

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


def mutate(genome: Genome, mutation_rate: float = 0.25) -> Genome:
    """Gaussian and uniform perturbations across all 28 genes."""
    g = copy.deepcopy(genome)
    g.name = f"Mut_{random.randint(1000, 9999)}"

    # Land
    if random.random() < mutation_rate:
        g.land_ne_min_day = max(4, min(7, g.land_ne_min_day + random.choice([-1, 0, 1])))
    if random.random() < mutation_rate:
        g.land_ne_min_money = max(900, min(1600, g.land_ne_min_money + random.choice([-100, 100])))
    if random.random() < mutation_rate:
        g.land_sw_min_day = max(9, min(13, g.land_sw_min_day + random.choice([-1, 0, 1])))
    if random.random() < mutation_rate:
        g.land_sw_min_money = max(2000, min(3000, g.land_sw_min_money + random.choice([-200, 200])))
    if random.random() < mutation_rate * 0.4:
        g.buy_se_land = not g.buy_se_land

    # Animals
    if random.random() < mutation_rate:
        g.target_cows = max(6, min(12, g.target_cows + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        g.target_sheep = max(2, min(6, g.target_sheep + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        g.animal_end_day = max(9, min(14, g.animal_end_day + random.choice([-1, 0, 1])))
    if random.random() < mutation_rate:
        g.wheat_reserve_mult = max(1, min(3, g.wheat_reserve_mult + random.choice([-1, 1])))

    # Strawberries
    if random.random() < mutation_rate:
        g.target_strawberries = max(30, min(50, g.target_strawberries + random.choice([-5, 5])))
    if random.random() < mutation_rate:
        g.strawberry_start_day = max(4, min(7, g.strawberry_start_day + random.choice([-1, 0, 1])))
    if random.random() < mutation_rate:
        g.strawberry_end_day = max(11, min(15, g.strawberry_end_day + random.choice([-1, 1])))

    # Labor
    if random.random() < mutation_rate:
        g.labor_early = max(5, min(8, g.labor_early + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        g.labor_mid = max(7, min(10, g.labor_mid + random.choice([-1, 1])))
    if random.random() < mutation_rate:
        g.labor_late = max(10, min(15, g.labor_late + random.choice([-1, 1])))

    # Pricing
    if random.random() < mutation_rate:
        g.price_throttle_ratio = round(max(0.55, min(0.85, g.price_throttle_ratio + random.choice([-0.05, 0.05]))), 2)
    if random.random() < mutation_rate:
        g.price_burst_ratio = round(max(1.05, min(1.25, g.price_burst_ratio + random.choice([-0.05, 0.05]))), 2)
    if random.random() < mutation_rate:
        g.batch_fertilizer = max(8, min(15, g.batch_fertilizer + random.choice([-2, 2])))
    if random.random() < mutation_rate:
        g.batch_milk = max(4, min(8, g.batch_milk + random.choice([-1, 1])))

    # Shop Synergy
    if random.random() < mutation_rate:
        g.shop_icecream_straw_boost = max(5, min(15, g.shop_icecream_straw_boost + random.choice([-5, 5])))
    if random.random() < mutation_rate:
        g.shop_yarn_sheep_boost = max(0, min(3, g.shop_yarn_sheep_boost + random.choice([-1, 1])))

    return g


def crossover(parent_a: Genome, parent_b: Genome) -> Genome:
    """Uniform genetic crossover combining traits from two fit parents."""
    dict_a = parent_a.to_dict()
    dict_b = parent_b.to_dict()
    child_dict = {}

    for key in dict_a:
        if key == "name":
            child_dict[key] = f"Cross_{random.randint(1000, 9999)}"
        else:
            child_dict[key] = dict_a[key] if random.random() < 0.5 else dict_b[key]

    return Genome.from_dict(child_dict)


class DeepGrandmasterAgent:
    """Execution engine featuring 28-gene brain, shop synergy, and collision prevention."""
    def __init__(self, genome: Genome):
        self.g = genome

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
        if day < self.g.strawberry_start_day:
            if money >= CROPS["MELON"]["seed"]:
                return "MELON"
            if money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Phase 2: Strawberry Engine
        if self.g.strawberry_start_day <= day <= self.g.strawberry_end_day:
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

        # Shop Synergies
        has_ice_cream_or_smoothie = any(s in ["ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "BRUNCH_SPOT"] for s in unlocked_shops)
        has_yarn_store = "YARN_STORE" in unlocked_shops

        eff_target_strawberries = self.g.target_strawberries + (self.g.shop_icecream_straw_boost if has_ice_cream_or_smoothie else 0)
        eff_target_sheep = self.g.target_sheep + (self.g.shop_yarn_sheep_boost if has_yarn_store else 0)

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
        wheat_reserve = (total_animals + animals_in_shed) * self.g.wheat_reserve_mult

        sell_schedule = [
            ("FERTILIZER", self.g.batch_fertilizer),
            ("MILK", self.g.batch_milk),
            ("WOOL", self.g.batch_wool),
            ("STRAWBERRY", self.g.batch_strawberry),
            ("MELON", self.g.batch_melon),
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

            if not is_endgame_liquidation and curr_price < base_p * self.g.price_throttle_ratio:
                effective_slice = max(1, base_slice // 2)
            elif curr_price >= base_p * self.g.price_burst_ratio:
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
            target_hires = self.g.labor_early
            if day >= self.g.labor_mid_day:
                target_hires = self.g.labor_mid
            if day >= self.g.labor_late_day:
                target_hires = self.g.labor_late

            while hires_today < target_hires and len(market_orders) < 9 and money >= 20:
                market_orders.append(["HIRE"])
                hires_today += 1
                money -= 20

        # 4. Gated Land Expansion
        if "NE" not in unlocked and money >= self.g.land_ne_min_money and day >= self.g.land_ne_min_day and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in unlocked and money >= self.g.land_sw_min_money and day >= self.g.land_sw_min_day and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 2000
        elif self.g.buy_se_land and "SE" not in unlocked and money >= self.g.land_se_min_money and day >= self.g.land_se_min_day and len(market_orders) < 9:
            market_orders.append(["BUY_LAND"])
            money -= 3000

        # 5. Balanced Animal Purchases
        total_capacity = len(PASTURE_LOCATIONS)
        if (total_animals + animals_in_shed) < total_capacity and day <= self.g.animal_end_day and len(market_orders) < 9:
            if total_sheep < eff_target_sheep and (total_cows >= total_sheep * self.g.cow_to_sheep_ratio or total_cows >= self.g.target_cows) and money >= 600:
                market_orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            elif total_cows < self.g.target_cows and money >= 500:
                market_orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400

        # 6. Wheat Feed Procurement
        needed_feed = (total_animals + animals_in_shed) * self.g.wheat_reserve_mult
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
                max_batch = self.g.strawberry_seed_batch if chosen_crop == "STRAWBERRY" else 6
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


def make_agent(genome: Genome):
    p_agent = DeepGrandmasterAgent(genome)
    def _agent(obs):
        return p_agent.act(obs)
    return _agent
