"""
Grandmaster Agent (4-Tier Modular Clean Architecture).
Generation: Grandmaster_v13_Titan

Architecture Overview:
- Tier 1: Perception & Domain Model (WorldState)
- Tier 2: Strategic Macroeconomic Engine (MacroEconomy)
- Tier 3: Tactical Unit Controller (UnitController)
- Tier 4: Orchestration Facade (GrandmasterAgent)

Evolved Parameters:
- NE Land: Day 6+ (Min Money: $1400)
- SW Land: Day 11+ (Min Money: $2700)
- Animals: 11 Cows, 3 Sheep (Cutoff Day 12)
- Dynamic Husbandry Squad: 2 units (<=5 animals), 3 units (<=9 animals), 4 units (10+ animals)
- Strawberries: Base 50 (Boost: +10 if IceCream/Smoothie shop opens)
- Sheep Synergy: +2 if Yarn Store opens
- Labor: Early 6, Mid 8 (Day 6), Late 11 (Day 10)
- Price Arbitrage: Throttle < 60%, Burst >= 120%
- Anti-Clog Shed Buffer: Shed >= 65 triggers burst liquidation (base + 4) of bulky goods
- Feed Reserve: 2x baseline, Day 29 smooth liquidation (1x)
- Anti-Oscillation Endgame: Disabled wheat pickups & watering in endgame churn
"""

from typing import Dict, List, Tuple, Optional, Set, Any

# ==============================================================================
# ENVIRONMENT CONSTANTS & GEOMETRY
# ==============================================================================

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


def _dist(a: Tuple[int, int], b: Tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _find_step(start_pos: Tuple[int, int], target_pos: Tuple[int, int], board_size: int = 10) -> Optional[str]:
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


def _get_quadrant(x: int, y: int, board_size: int = 10) -> str:
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


# ==============================================================================
# TIER 1: PERCEPTION & DOMAIN MODEL
# ==============================================================================

class WorldState:
    """Perception layer: encapsulates and structures raw Kaggle observations."""
    def __init__(self, obs: Dict[str, Any]):
        self.raw = obs
        self.player: int = obs["player"]
        self.farms: List[Dict[str, Any]] = obs["farms"]
        self.farm: Dict[str, Any] = self.farms[self.player]
        self.private: Dict[str, Any] = obs["private"]
        
        self.day: int = obs.get("day", 0)
        self.hour: int = obs.get("hour", 0)
        self.step: int = obs.get("step", 0)
        self.remaining_days: int = 30 - self.day
        self.remaining_steps: int = 720 - self.step
        
        self.money: int = self.farm["money"]
        self.shed: Dict[str, int] = self.private.get("shed", {})
        self.seeds: Dict[str, int] = self.private.get("seeds", {})
        self.inventories: List[Dict[str, int]] = self.private.get("inventories", [])
        self.unlocked: Set[str] = set(self.farm.get("unlocked_quadrants", ["NW"]))
        
        self.tiles: List[List[Any]] = self.farm["tiles"]
        self.board_size: int = len(self.tiles)
        
        self.market_prices: Dict[str, int] = obs.get("market", {}).get("prices", {})
        self.unlocked_shops: List[str] = obs.get("town", {}).get("unlocked_shops", [])
        
        # Town demand synergy flags
        self.has_ice_cream_or_smoothie: bool = any(
            s in ["ICE_CREAM_SHOP", "SMOOTHIE_SHOP", "BRUNCH_SPOT"] for s in self.unlocked_shops
        )
        self.has_yarn_store: bool = "YARN_STORE" in self.unlocked_shops
        self.target_strawberries: int = 50 + (10 if self.has_ice_cream_or_smoothie else 0)
        self.target_sheep: int = 3 + (2 if self.has_yarn_store else 0)
        
        # Herd metrics
        self.total_cows: int = sum(1 for row in self.tiles for t in row if isinstance(t, dict) and t.get("animal") == "COW")
        self.total_sheep: int = sum(1 for row in self.tiles for t in row if isinstance(t, dict) and t.get("animal") == "SHEEP")
        self.total_animals: int = self.total_cows + self.total_sheep
        self.animals_in_shed: int = self.shed.get("COW", 0) + self.shed.get("SHEEP", 0)
        self.strawberry_count: int = sum(1 for row in self.tiles for t in row if isinstance(t, dict) and t.get("crop") == "STRAWBERRY")
        
        # Shed metrics & Anti-Clog
        self.shed_total: int = sum(self.shed.values())
        self.shed_is_crowded: bool = self.shed_total >= 65
        
        # Endgame state
        self.is_endgame: bool = (self.day >= 29 and self.hour >= 16) or (self.remaining_steps <= 18)
        self.is_endgame_liquidation: bool = (self.day >= 29 and self.hour >= 16) or (self.remaining_steps <= 12)


# ==============================================================================
# TIER 2: STRATEGIC MACROECONOMIC ENGINE
# ==============================================================================

class MacroEconomy:
    """Strategic Economic Brain: manages market orders, liquidity, assets, and trade."""
    
    @staticmethod
    def select_crop(day: int, remaining_days: int, money: int, strawberry_count: int, target_strawberries: int) -> Optional[str]:
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
        if 5 <= day <= 12:
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

    def plan_market_orders(self, state: WorldState) -> List[List[Any]]:
        orders: List[List[Any]] = []
        money = state.money

        # 1. Day 0 Master Opening
        if state.step == 0 or (state.step == 1 and len(state.unlocked) == 1 and money >= 2500):
            return [
                ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"], ["HIRE"],
                ["BUY_ANIMAL", "COW", 2],
                ["BUY_ANIMAL", "SHEEP", 2],
                ["BUY_SEED", "MELON", 12],
                ["BUY_SEED", "WHEAT", 7],
                ["BUY_PRODUCT", "WHEAT", 6],
            ][:10]

        # 2. Price-Aware Sales with Anti-Clog Shed Protection
        reserve_mult = 1 if state.day >= 29 else 2
        wheat_reserve = (state.total_animals + state.animals_in_shed) * reserve_mult

        sell_schedule = [
            ("FERTILIZER", 8),
            ("MILK", 6),
            ("WOOL", 4),
            ("STRAWBERRY", 8),
            ("MELON", 12 if state.shed.get("MELON", 0) >= 20 else 6),
            ("WHEAT", 15),
            ("CARROT", 15),
            ("TOMATO", 10),
            ("EGG", 10),
        ]

        for prod, base_slice in sell_schedule:
            qty = state.shed.get(prod, 0)
            if qty <= 0 or len(orders) >= 9:
                continue

            curr_price = state.market_prices.get(prod, BASE_PRICES.get(prod, 50))
            base_p = BASE_PRICES.get(prod, 50)

            if state.shed_is_crowded and prod in ("MELON", "FERTILIZER", "STRAWBERRY", "MILK", "WOOL"):
                effective_slice = max(base_slice, base_slice + 4)
            elif not state.is_endgame_liquidation and curr_price < base_p * 0.6:
                effective_slice = max(1, base_slice // 2)
            elif curr_price >= base_p * 1.2:
                effective_slice = base_slice + 2
            else:
                effective_slice = base_slice

            if prod == "WHEAT" and not state.is_endgame_liquidation:
                sellable_wheat = max(0, qty - wheat_reserve)
                if sellable_wheat > 0:
                    orders.append(["SELL", "WHEAT", min(sellable_wheat, effective_slice)])
                continue

            sell_qty = qty if state.is_endgame_liquidation else min(qty, effective_slice)
            orders.append(["SELL", prod, sell_qty])

        # 3. Daily Labor Re-Hire
        hires_today = state.farm.get("hires_today", 0)
        if (state.hour == 1 or state.hour == 2) and state.remaining_days > 2:
            target_hires = 6
            if state.day >= 6:
                target_hires = 8
            if state.day >= 10:
                target_hires = 11

            while hires_today < target_hires and len(orders) < 9 and money >= 20:
                orders.append(["HIRE"])
                hires_today += 1
                money -= 20

        # 4. Gated Land Expansion
        if "NE" not in state.unlocked and money >= 1400 and state.day >= 6 and len(orders) < 9:
            orders.append(["BUY_LAND"])
            money -= 1000
        elif "SW" not in state.unlocked and money >= 2700 and state.day >= 11 and len(orders) < 9:
            orders.append(["BUY_LAND"])
            money -= 2000

        # 5. Balanced Animal Purchases
        total_capacity = len(PASTURE_LOCATIONS)
        if (state.total_animals + state.animals_in_shed) < total_capacity and state.day <= 12 and len(orders) < 9:
            if state.total_sheep < state.target_sheep and (state.total_cows >= state.total_sheep * 2.5 or state.total_cows >= 11) and money >= 600:
                orders.append(["BUY_ANIMAL", "SHEEP", 1])
                money -= 500
            elif state.total_cows < 11 and money >= 500:
                orders.append(["BUY_ANIMAL", "COW", 1])
                money -= 400

        # 6. Wheat Feed Procurement
        needed_feed = (state.total_animals + state.animals_in_shed) * reserve_mult
        current_wheat = state.shed.get("WHEAT", 0)
        if not state.is_endgame_liquidation and state.day < 29 and (state.total_animals + state.animals_in_shed) > 0 and current_wheat < needed_feed and money >= 100 and len(orders) < 9:
            buy_wheat_amt = min(8, needed_feed - current_wheat + 2)
            if buy_wheat_amt > 0:
                orders.append(["BUY_PRODUCT", "WHEAT", buy_wheat_amt])
                money -= buy_wheat_amt * 25

        # 7. Seed Purchases
        chosen_crop = self.select_crop(state.day, state.remaining_days, money, state.strawberry_count, state.target_strawberries)
        if not state.is_endgame_liquidation and state.day < 28 and chosen_crop and state.remaining_days > 2 and len(orders) < 9:
            current_count = state.seeds.get(chosen_crop, 0)
            max_cap = 25 if chosen_crop == "STRAWBERRY" else 8
            if current_count < max_cap:
                seed_cost = CROPS[chosen_crop]["seed"]
                max_batch = 10 if chosen_crop == "STRAWBERRY" else 6
                buy_n = min(max_batch, int(money // seed_cost))
                if buy_n > 0:
                    orders.append(["BUY_SEED", chosen_crop, buy_n])
                    money -= buy_n * seed_cost

        return orders[:10]


# ==============================================================================
# TIER 3: TACTICAL UNIT CONTROLLER
# ==============================================================================

class UnitController:
    """Tactical Unit Controller: plans atomic movements and interactions for individual workers."""
    
    @staticmethod
    def plan_unit_action(
        unit_idx: int,
        unit_pos: List[int],
        role: str,
        state: WorldState,
        claimed_tiles: Set[Tuple[int, int]],
        shared_state: Dict[str, Any],
    ) -> Tuple[List[str], Tuple[int, int]]:
        ux, uy = unit_pos
        board_size = state.board_size
        tile = state.tiles[uy][ux]
        inv = state.inventories[unit_idx] if unit_idx < len(state.inventories) else {}
        shed = state.shed
        seeds = state.seeds
        is_endgame = state.is_endgame
        unlocked = state.unlocked

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

            if not is_endgame and role == "HUSBANDRY" and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                has_unfed = any(
                    isinstance(state.tiles[y][x], dict)
                    and "animal" in state.tiles[y][x]
                    and not state.tiles[y][x].get("fed_today", False)
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfed:
                    take_amt = min(4, shed.get("WHEAT", 0))
                    return ["PICKUP", "WHEAT", take_amt], (ux, uy)

            if role == "HUSBANDRY" and not carrying_animal and not shared_state["animal_pickup_claimed"]:
                for px, py in PASTURE_LOCATIONS:
                    if _get_quadrant(px, py, board_size) in unlocked:
                        pt = state.tiles[py][px]
                        if pt is None or (isinstance(pt, dict) and pt.get("kind") == "PASTURE" and "animal" not in pt):
                            if shed.get("COW", 0) > 0:
                                shared_state["animal_pickup_claimed"] = True
                                return ["PICKUP", "COW", 1], (ux, uy)
                            if shed.get("SHEEP", 0) > 0:
                                shared_state["animal_pickup_claimed"] = True
                                return ["PICKUP", "SHEEP", 1], (ux, uy)

            if role == "AGRICULTURE" and not is_endgame and state.day <= 26 and inv.get("FERTILIZER", 0) == 0 and shed.get("FERTILIZER", 0) > 0:
                has_unfert = any(
                    isinstance(state.tiles[y][x], dict)
                    and state.tiles[y][x].get("crop") in ("MELON", "STRAWBERRY")
                    and state.tiles[y][x].get("fertilized_until_day", -1) < state.day
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfert:
                    take_fert = min(2, shed.get("FERTILIZER", 0))
                    return ["PICKUP", "FERTILIZER", take_fert], (ux, uy)

        # --- 2. Current Tile Actions ---
        if isinstance(tile, dict):
            if "animal" in tile:
                if is_endgame and tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if not is_endgame and not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not is_endgame and not tile.get("cared_today", False):
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
                age = state.day - tile.get("planted_day", 0)
                c_data = CROPS.get(crop, {})
                peak = age >= c_data.get("max_yield_day", 99)
                if (peak or is_endgame) and tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if crop in ("MELON", "STRAWBERRY") and inv.get("FERTILIZER", 0) > 0 and tile.get("fertilized_until_day", -1) < state.day:
                    return ["FERTILIZE"], (ux, uy)
                if not is_endgame and not tile.get("watered_today", False):
                    return ["WATER"], (ux, uy)

            if tile.get("kind") == "WEED":
                return ["DIG"], (ux, uy)

        if (ux, uy) in PASTURE_LOCATIONS and _get_quadrant(ux, uy, board_size) in unlocked:
            if tile is None and (carrying_animal or role == "HUSBANDRY"):
                return ["BUILD_PASTURE"], (ux, uy)

        if tile is None and (ux, uy) not in claimed_tiles and (ux, uy) not in PASTURE_LOCATIONS:
            if _get_quadrant(ux, uy, board_size) in unlocked and not is_endgame:
                best_crop = MacroEconomy.select_crop(state.day, state.remaining_days, state.money, shared_state.get("strawberries", 0), state.target_strawberries)
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
                        pt = state.tiles[py][px]
                        if pt is None or (isinstance(pt, dict) and pt.get("kind") == "PASTURE" and "animal" not in pt):
                            d = _dist((ux, uy), (px, py))
                            if d < best_dist:
                                best_dist = d
                                best_target = (px, py)
                if best_target:
                    step_dir = _find_step((ux, uy), best_target, board_size)
                    if step_dir:
                        return [step_dir], best_target

            if not is_endgame:
                has_unfed = any(
                    isinstance(state.tiles[y][x], dict)
                    and "animal" in state.tiles[y][x]
                    and not state.tiles[y][x].get("fed_today", False)
                    for y in range(board_size) for x in range(board_size)
                )
                if has_unfed and inv.get("WHEAT", 0) == 0 and shed.get("WHEAT", 0) > 0:
                    step_to_shed = _find_step((ux, uy), SHED_CENTER, board_size)
                    if step_to_shed:
                        return [step_to_shed], SHED_CENTER

            for y in range(board_size):
                for x in range(board_size):
                    t = state.tiles[y][x]
                    if isinstance(t, dict) and "animal" in t:
                        if (x, y) in claimed_tiles:
                            continue
                        prio = -1
                        if t.get("yield_units", 0) > 0:
                            prio = 10
                        elif not is_endgame and not t.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                            prio = 9
                        elif not is_endgame and not t.get("cared_today", False):
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
                    t = state.tiles[y][x]
                    if t == "LOCKED":
                        continue

                    d = _dist((ux, uy), (x, y))
                    prio = -1

                    if isinstance(t, dict) and t.get("kind") == "PLANT":
                        c_data = CROPS.get(t["crop"], {})
                        age = state.day - t["planted_day"]
                        if (age >= c_data["max_yield_day"] or is_endgame) and t.get("yield_units", 0) > 0:
                            prio = 10
                        elif not is_endgame and not t.get("watered_today", False):
                            prio = 8
                        elif not is_endgame and t.get("crop") in ("MELON", "STRAWBERRY") and inv.get("FERTILIZER", 0) > 0 and t.get("fertilized_until_day", -1) < state.day:
                            prio = 7
                    elif not is_endgame and t is None and (x, y) not in PASTURE_LOCATIONS and state.remaining_days > 1:
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


# ==============================================================================
# TIER 4: ORCHESTRATION FACADE & KAGGLE ENTRY POINT
# ==============================================================================

class GrandmasterAgent:
    """Decoupled, modular orchestrator connecting WorldState, MacroEconomy, and UnitController."""
    def __init__(self):
        self.economy = MacroEconomy()
        self.controller = UnitController()

    def select_crop(self, day: int, remaining_days: int, money: int, strawberry_count: int, target_strawberries: int) -> Optional[str]:
        return self.economy.select_crop(day, remaining_days, money, strawberry_count, target_strawberries)

    def plan_unit_turn(self, unit_idx, unit_pos, role, farm, private, day, hour, step, claimed_tiles, shared_state, target_strawberries):
        obs = {
            "player": 0,
            "farms": [farm, farm],
            "private": private,
            "day": day,
            "hour": hour,
            "step": step,
            "market": {"prices": {}},
            "town": {"unlocked_shops": []},
        }
        state = WorldState(obs)
        state.target_strawberries = target_strawberries
        return self.controller.plan_unit_action(unit_idx, unit_pos, role, state, claimed_tiles, shared_state)

    def act(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        state = WorldState(obs)
        
        # 1. Day 0 Master Opening & Economic orders
        if state.step == 0 or (state.step == 1 and len(state.unlocked) == 1 and state.money >= 2500):
            return {
                "farmer": ["BUILD_PASTURE"],
                "hands": [],
                "market": self.economy.plan_market_orders(state),
            }

        market_orders = self.economy.plan_market_orders(state)

        # 2. Squad Specialization Execution
        claimed_tiles: Set[Tuple[int, int]] = set()
        shared_state: Dict[str, Any] = {
            "animal_pickup_claimed": False,
            "strawberries": state.strawberry_count,
        }

        husbandry_count = 2 if state.total_animals <= 5 else (3 if state.total_animals <= 9 else 4)

        # Farmer action
        farmer_role = "HUSBANDRY"
        farmer_act, _ = self.controller.plan_unit_action(
            0, state.farm["farmer"], farmer_role, state, claimed_tiles, shared_state
        )

        # Hands action
        hands_acts = []
        for h_idx, h_pos in enumerate(state.farm.get("hands", [])):
            unit_id = h_idx + 1
            h_role = "HUSBANDRY" if unit_id < husbandry_count else "AGRICULTURE"
            h_act, _ = self.controller.plan_unit_action(
                unit_id, h_pos, h_role, state, claimed_tiles, shared_state
            )
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders,
        }


# Backwards compatibility alias
DeepChampionAgent = GrandmasterAgent

_champion_brain = GrandmasterAgent()


def agent(obs: Dict[str, Any]) -> Dict[str, Any]:
    """Kaggle Environments Entry Point."""
    return _champion_brain.act(obs)
