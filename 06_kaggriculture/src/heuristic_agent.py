"""
Advanced Heuristic Strategy Agent for Kaggriculture.
Features:
- Multi-crop portfolio management (Carrot + Wheat + high value crops)
- Dynamic town shop demand awareness
- Efficient multi-unit task allocation (Farmer + Hired Hands)
- Spatial pathfinding & watering / harvesting coverage
- Land expansion (unlocking NE quadrant when profitable)
- End-of-season countdown & full shed liquidation before turn 720
"""

from collections import deque

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]

MOVES = {
    (0, -1): "NORTH",
    (0, 1): "SOUTH",
    (1, 0): "EAST",
    (-1, 0): "WEST",
}


def get_quadrant(x, y, board_size=10):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def find_path_step(start_pos, target_pos, board_size=10):
    """
    Returns the first move direction from start_pos to target_pos using BFS.
    """
    sx, sy = start_pos
    tx, ty = target_pos
    if (sx, sy) == (tx, ty):
        return None

    # Greedy Manhattan first if path is clear
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


class StrategicAgent:
    def __init__(self):
        self.preferred_crop = "CARROT"

    def select_best_crop_to_plant(self, day, remaining_days, money, town_shops, market_prices):
        """
        Determines the optimal crop given days remaining, cash, and market prices.
        """
        if remaining_days <= 1:
            return None  # No time for anything
        if remaining_days <= 3:
            # Only Carrot or Wheat can mature
            if remaining_days >= 3 and money >= CROPS["CARROT"]["seed"]:
                return "CARROT"
            if remaining_days >= 2 and money >= CROPS["WHEAT"]["seed"]:
                return "WHEAT"
            return None

        # Middle or early game: evaluate profitability
        # Check town demand
        carrot_demand = sum(1 for s in town_shops if s in ("PET_CAFE", "FARMERS_MARKET"))
        wheat_demand = sum(1 for s in town_shops if s in ("BAKERY", "PIZZA_SHOP", "BRUNCH_SPOT", "ICE_CREAM_SHOP", "FARMERS_MARKET"))

        carrot_price = market_prices.get("CARROT", 35)
        wheat_price = market_prices.get("WHEAT", 25)

        # Early days (day 0-16): Carrot is high throughput, fast rotation
        if carrot_price >= 20 and money >= CROPS["CARROT"]["seed"]:
            return "CARROT"
        elif wheat_price >= 15 and money >= CROPS["WHEAT"]["seed"]:
            return "WHEAT"
        elif money >= CROPS["CARROT"]["seed"]:
            return "CARROT"
        return None

    def plan_unit_action(self, unit_pos, farm, private, day, hour, remaining_steps, chosen_crop, claimed_tiles):
        """
        Decides action for a single farmer or hired hand.
        """
        ux, uy = unit_pos
        board_size = len(farm["tiles"])
        tile = farm["tiles"][uy][ux]
        seeds = private.get("seeds", {})
        money = farm.get("money", 0)

        # 1. Action on current tile:
        # A) Harvest ready crops
        if isinstance(tile, dict) and tile.get("kind") == "PLANT":
            crop_name = tile["crop"]
            crop_data = CROPS[crop_name]
            age = day - tile["planted_day"]
            yield_units = tile.get("yield_units", 0)

            # Harvest if at peak or close to decay or near season end
            is_peak = age >= crop_data["max_yield_day"]
            is_endgame = remaining_steps <= 24

            if (is_peak or is_endgame) and yield_units > 0:
                if not crop_data["ongoing"] or age >= crop_data["first_yield_day"]:
                    return ["HARVEST"], (ux, uy)

            # Water if not watered today
            if not tile.get("watered_today", False):
                return ["WATER"], (ux, uy)

        # B) Plant on current tile if empty and owned
        if tile is None and (ux, uy) not in claimed_tiles:
            if chosen_crop and seeds.get(chosen_crop, 0) > 0:
                claimed_tiles.add((ux, uy))
                return ["PLANT", chosen_crop], (ux, uy)

        # C) Dig weeds on current tile
        if isinstance(tile, dict) and tile.get("kind") == "WEED":
            return ["DIG"], (ux, uy)

        # 2. Seek target tile across accessible quadrants
        # Find all tiles that need WATER, HARVEST, DIG, or PLANTING
        best_target = None
        best_dist = 9999
        best_priority = -1

        unlocked_quads = set(farm.get("unlocked_quadrants", ["NW"]))

        for y in range(board_size):
            for x in range(board_size):
                if (x, y) in claimed_tiles:
                    continue
                q = get_quadrant(x, y, board_size)
                if q not in unlocked_quads:
                    continue
                t = farm["tiles"][y][x]
                if t == "LOCKED":
                    continue

                dist = abs(x - ux) + abs(y - uy)
                priority = -1

                if isinstance(t, dict) and t.get("kind") == "PLANT":
                    c_data = CROPS[t["crop"]]
                    p_age = day - t["planted_day"]
                    y_units = t.get("yield_units", 0)
                    if (p_age >= c_data["max_yield_day"] or remaining_steps <= 24) and y_units > 0:
                        priority = 4  # Highest priority: Harvest!
                    elif not t.get("watered_today", False):
                        priority = 3  # High priority: Water!
                elif isinstance(t, dict) and t.get("kind") == "WEED":
                    priority = 1  # Clear weed
                elif t is None and chosen_crop and seeds.get(chosen_crop, 0) > len(claimed_tiles):
                    priority = 2  # Empty space to plant

                if priority > best_priority or (priority == best_priority and dist < best_dist):
                    best_priority = priority
                    best_dist = dist
                    best_target = (x, y)

        if best_target is not None and best_target != (ux, uy):
            step_dir = find_path_step((ux, uy), best_target, board_size)
            if step_dir:
                claimed_tiles.add(best_target)
                return [step_dir], best_target

        return ["PASS"], (ux, uy)

    def __call__(self, obs):
        player = obs["player"]
        farms = obs["farms"]
        farm = farms[player]
        private = obs["private"]
        day = obs.get("day", 0)
        hour = obs.get("hour", 0)
        step = obs.get("step", 0)
        total_steps = 720
        remaining_steps = total_steps - step
        remaining_days = 30 - day

        money = farm["money"]
        market_obs = obs.get("market", {})
        market_prices = market_obs.get("prices", {})
        town = obs.get("town", {})
        town_shops = town.get("unlocked_shops", [])

        market_orders = []

        # 1. Market Operations:
        # A) Sell any produce sitting in the shed every single turn
        shed = private.get("shed", {})
        for prod in ["CARROT", "WHEAT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]:
            qty = shed.get(prod, 0)
            if qty > 0 and len(market_orders) < 8:
                market_orders.append(["SELL", prod, qty])

        # B) Land expansion: Unlock NE quadrant ($1000) when we have ample savings (> $1400)
        unlocked = farm.get("unlocked_quadrants", ["NW"])
        if "NE" not in unlocked and money >= 1400 and day <= 20:
            market_orders.append(["BUY_LAND"])
            money -= 1000

        # C) Hiring farm hands:
        # On hour 0 of each day, hire 1 hand if we have > 15 tiles unlocked and balance > $200
        # Hire 2nd hand if money > $800
        hires_today = farm.get("hires_today", 0)
        if hour == 0 and remaining_days > 2:
            if hires_today == 0 and money >= 100:
                market_orders.append(["HIRE"])
                money -= 1
            elif hires_today == 1 and money >= 500:
                market_orders.append(["HIRE"])
                money -= 1

        # D) Seed procurement:
        chosen_crop = self.select_best_crop_to_plant(day, remaining_days, money, town_shops, market_prices)
        current_seeds = private.get("seeds", {}).get(chosen_crop, 0) if chosen_crop else 0

        # Count empty unlocked tiles
        empty_count = 0
        board_size = len(farm["tiles"])
        for y in range(board_size):
            for x in range(board_size):
                if farm["tiles"][y][x] is None:
                    empty_count += 1

        needed_seeds = max(0, min(empty_count, 15) - current_seeds)
        if chosen_crop and needed_seeds > 0 and remaining_days > 2:
            seed_price = CROPS[chosen_crop]["seed"]
            max_buy = int(money // seed_price)
            buy_qty = min(needed_seeds, max_buy, 5)
            if buy_qty > 0 and len(market_orders) < 10:
                market_orders.append(["BUY_SEED", chosen_crop, buy_qty])
                money -= buy_qty * seed_price

        # 2. Units Operations (Farmer & Hands):
        claimed_tiles = set()
        farmer_act, _ = self.plan_unit_action(
            farm["farmer"], farm, private, day, hour, remaining_steps, chosen_crop, claimed_tiles
        )

        hands_acts = []
        for hand_pos in farm.get("hands", []):
            h_act, _ = self.plan_unit_action(
                hand_pos, farm, private, day, hour, remaining_steps, chosen_crop, claimed_tiles
            )
            hands_acts.append(h_act)

        return {
            "farmer": farmer_act,
            "hands": hands_acts,
            "market": market_orders[:10],
        }


# Global instance for execution
_strategic_agent_instance = StrategicAgent()


def heuristic_agent(obs):
    return _strategic_agent_instance(obs)
