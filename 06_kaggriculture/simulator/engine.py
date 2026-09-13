"""
Pure Python standalone simulator for Kaggriculture.
Matches official kaggle-environments rules exactly, without external dependencies.
"""

import copy
import json
import math
import random

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}

ANIMALS = {
    "GOOSE": {"cost": 300, "structure": "COOP",    "first_yield_day": 4, "interval": 1, "max_held": 4, "product": "EGG"},
    "COW":   {"cost": 400, "structure": "PASTURE", "first_yield_day": 8, "interval": 2, "max_held": 6, "product": "MILK"},
    "SHEEP": {"cost": 500, "structure": "PASTURE", "first_yield_day": 6, "interval": 3, "max_held": 6, "product": "WOOL"},
}

PRODUCTS = ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON", "EGG", "MILK", "WOOL", "FERTILIZER"]

MARKET_I0 = 10000
PRICE_FLOOR = 1

MARKET_PARAMS = {
    "WHEAT":      {"base":  25, "I0": MARKET_I0, "T": 400, "below_func": "sqrt",   "below_target": 0.80, "above_func": "log",    "above_target": 0.20},
    "CARROT":     {"base":  35, "I0": MARKET_I0, "T": 450, "below_func": "hinge",  "below_target": 1.00, "above_func": "sqrt",   "above_target": 0.70},
    "TOMATO":     {"base":  60, "I0": MARKET_I0, "T": 200, "below_func": "hinge",  "below_target": 0.40, "above_func": "sqrt",   "above_target": 0.60},
    "STRAWBERRY": {"base": 120, "I0": MARKET_I0, "T": 100, "below_func": "sqrt",   "below_target": 0.70, "above_func": "linear", "above_target": 1.60},
    "MELON":      {"base": 250, "I0": MARKET_I0, "T": 300, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.60},
    "EGG":        {"base":  50, "I0": MARKET_I0, "T": 332, "below_func": "hinge",  "below_target": 0.40, "above_func": "log",    "above_target": 0.20},
    "MILK":       {"base": 160, "I0": MARKET_I0, "T": 122, "below_func": "sqrt",   "below_target": 0.60, "above_func": "linear", "above_target": 1.60},
    "WOOL":       {"base": 200, "I0": MARKET_I0, "T": 105, "below_func": "log",    "below_target": 0.20, "above_func": "sq",     "above_target": 3.20},
    "FERTILIZER": {"base": 100, "I0": MARKET_I0, "T": 200, "below_func": "linear", "below_target": 0.40, "above_func": "linear", "above_target": 0.40},
}

HINGE_GAIN = 8.0

FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST":  (1, 0),
    "WEST":  (-1, 0),
}

LAND_ORDER = ["NE", "SW", "SE"]
LAND_PRICES = [1000, 2000, 4000]
FARM_HAND_COST_MULT = 1

SHOPS = {
    "BAKERY":         ["EGG", "WHEAT"],
    "PIZZA_SHOP":     ["MILK", "TOMATO", "WHEAT"],
    "BRUNCH_SPOT":    ["EGG", "WHEAT", "STRAWBERRY"],
    "YARN_STORE":     ["WOOL"],
    "ICE_CREAM_SHOP": ["STRAWBERRY", "MILK", "WHEAT"],
    "PET_CAFE":       ["CARROT"],
    "SMOOTHIE_SHOP":  ["STRAWBERRY", "MILK"],
    "FARMERS_MARKET": ["WHEAT", "CARROT", "TOMATO", "STRAWBERRY"],
}

TOWN_CENTER_PRODUCTS = [p for p in PRODUCTS if p != "FERTILIZER"]
MAX_SHOP_INSTANCES = 8


def _shape(func, x, T=None):
    x = max(0.0, float(x))
    if func == "linear": return x
    if func == "sq":     return x * x
    if func == "sqrt":   return math.sqrt(x)
    if func == "log":    return math.log(1.0 + x)
    if func == "log10":  return math.log10(1.0 + x)
    if func == "hinge":
        if not T or T <= 0:
            return x
        u = x / T
        return u + HINGE_GAIN * max(0.0, u - 1.0) ** 2
    return x


def market_price(item, inventory, params=None):
    p = (params or MARKET_PARAMS)[item]
    base = p["base"]
    I0 = p["I0"]
    T = p["T"]
    if inventory < I0:
        f = p["below_func"]
        amp = p["below_target"] * base / _shape(f, T, T)
        price = base + amp * _shape(f, I0 - inventory, T)
    else:
        f = p["above_func"]
        amp = p["above_target"] * base / _shape(f, T, T)
        price = base - amp * _shape(f, inventory - I0, T)
    return max(PRICE_FLOOR, int(round(price)))


def _refresh_prices(market):
    params = market.get("params")
    for item in PRODUCTS:
        market["prices"][item] = market_price(item, market["inventory"][item], params)


def _quadrant_of(x, y, board_size):
    half = board_size // 2
    return ("N" if y < half else "S") + ("W" if x < half else "E")


def _shed_access_tiles(board_size):
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _is_shed_adjacent(pos, board_size):
    return tuple(pos) in set(_shed_access_tiles(board_size))


def _initial_tile(x, y, board_size):
    return None if _quadrant_of(x, y, board_size) == "NW" else "LOCKED"


def _default_spawn(board_size):
    for tile in _shed_access_tiles(board_size):
        if _quadrant_of(tile[0], tile[1], board_size) == "NW":
            return tile
    return (0, 0)


def _spawn_hand(farm, board_size):
    occupants = {tile: 0 for tile in _shed_access_tiles(board_size)}
    all_pos = [tuple(farm["farmer"])] + [tuple(p) for p in farm["hands"]]
    for pos in all_pos:
        if pos in occupants:
            occupants[pos] += 1
    best = sorted(occupants.items(), key=lambda kv: (kv[1], _shed_access_tiles(board_size).index(kv[0])))
    return list(best[0][0])


def _fib(n):
    a, b = 1, 1
    for _ in range(n):
        a, b = b, a + b
    return a


class KaggricultureEnv:
    """
    Simulation Environment for Kaggriculture.
    Supports headless stepping and full matches between 2 agents.
    """
    def __init__(self, config=None, seed=None):
        self.config = {
            "episodeSteps": 720,
            "boardSize": 10,
            "startingMoney": 3000,
            "maxMarketOrdersPerTurn": 10,
            "turnsPerDay": 24,
            "shedCapacity": 100,
            "weedSpawnChance": 0.005,
            "townShopUnlockInterval": 3,
            "townShopSellInterval": 4,
            "townCenterSellInterval": 24,
            "farmHandCostMult": 1,
        }
        if config:
            self.config.update(config)

        self.seed = seed if seed is not None else random.randint(1, 1_000_000)
        self.rng = random.Random(self.seed)
        self.step_count = 0
        self.done = False

        board_size = self.config["boardSize"]
        starting_money = self.config["startingMoney"]

        self.farms = [
            {
                "money": float(starting_money),
                "tiles": [[_initial_tile(x, y, board_size) for x in range(board_size)] for y in range(board_size)],
                "farmer": list(_default_spawn(board_size)),
                "hands": [],
                "unlocked_quadrants": ["NW"],
                "hires_today": 0,
            }
            for _ in range(2)
        ]

        self.privates = [
            {
                "shed": {item: 0 for item in PRODUCTS + list(ANIMALS)},
                "seeds": {crop: 0 for crop in CROPS},
                "inventories": [{}],
            }
            for _ in range(2)
        ]

        self.market = {
            "inventory": {item: MARKET_I0 for item in PRODUCTS},
            "prices": {item: MARKET_PARAMS[item]["base"] for item in PRODUCTS},
        }
        self.town = {"unlocked_shops": []}

    @property
    def day(self):
        return self.step_count // self.config["turnsPerDay"]

    @property
    def hour(self):
        return self.step_count % self.config["turnsPerDay"]

    def get_observation(self, player_id):
        return {
            "player": player_id,
            "step": self.step_count,
            "day": self.day,
            "hour": self.hour,
            "farms": copy.deepcopy(self.farms),
            "market": copy.deepcopy(self.market),
            "town": copy.deepcopy(self.town),
            "private": copy.deepcopy(self.privates[player_id]),
        }

    def step(self, actions):
        """
        Execute one turn.
        actions: list of [action_p0, action_p1]
        """
        if self.done:
            return

        board_size = self.config["boardSize"]
        turns_per_day = self.config["turnsPerDay"]
        shed_capacity = self.config["shedCapacity"]
        day = self.day

        # 1. Apply farmer and hands actions
        for i in range(2):
            act = actions[i] if i < len(actions) and isinstance(actions[i], dict) else {}
            farmer_act = act.get("farmer", ["PASS"]) or ["PASS"]
            hands_act = act.get("hands", []) or []

            unit_actions = [farmer_act, *hands_act]
            plant_demand = {}
            for a in unit_actions:
                if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT":
                    plant_demand[a[1]] = plant_demand.get(a[1], 0) + 1
            seeds = self.privates[i]["seeds"]
            blocked = {crop for crop, n in plant_demand.items() if n > seeds.get(crop, 0)}

            def _allowed(a):
                if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] in blocked:
                    return ["PASS"]
                return a

            self._apply_unit_action(i, 0, _allowed(farmer_act))
            for h_idx, h_action in enumerate(hands_act):
                self._apply_unit_action(i, h_idx + 1, _allowed(h_action))

        # 2. Process market orders
        self._process_market(actions)

        # 3. Town consumption
        self._town_consume()

        # 4. Decay plants
        for farm in self.farms:
            self._decay_plants(farm)

        # 5. End of day check
        if (self.step_count + 1) % turns_per_day == 0:
            self._end_of_day()

        self.step_count += 1
        if self.step_count >= self.config["episodeSteps"]:
            self.done = True

    def _apply_unit_action(self, player_id, unit_idx, action):
        if not isinstance(action, list) or not action:
            return
        farm = self.farms[player_id]
        private = self.privates[player_id]
        board_size = self.config["boardSize"]
        day = self.day
        turns_per_day = self.config["turnsPerDay"]
        shed_capacity = self.config["shedCapacity"]

        op = action[0]
        pos = farm["farmer"] if unit_idx == 0 else (farm["hands"][unit_idx - 1] if unit_idx - 1 < len(farm["hands"]) else None)
        if pos is None:
            return
        fx, fy = pos[0], pos[1]

        while len(private["inventories"]) <= unit_idx:
            private["inventories"].append({})
        inv = private["inventories"][unit_idx]

        if op in FARMER_MOVES:
            dx, dy = FARMER_MOVES[op]
            nx, ny = fx + dx, fy + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if unit_idx == 0:
                    farm["farmer"] = [nx, ny]
                else:
                    farm["hands"][unit_idx - 1] = [nx, ny]
            return

        if op == "PASS":
            return

        tile = farm["tiles"][fy][fx]

        if op == "DROP":
            if not _is_shed_adjacent((fx, fy), board_size):
                return
            shed = private["shed"]
            for item, n in list(inv.items()):
                if n <= 0:
                    del inv[item]
                    continue
                room = max(0, shed_capacity - sum(shed.values()))
                take = min(n, room)
                if take > 0:
                    shed[item] = shed.get(item, 0) + take
                del inv[item]
            return

        if op == "PICKUP":
            if not _is_shed_adjacent((fx, fy), board_size):
                return
            if len(action) < 2:
                return
            item = action[1]
            n = int(action[2]) if len(action) >= 3 else 1
            if n <= 0:
                return
            available = private["shed"].get(item, 0)
            take = min(n, available)
            if take <= 0:
                return
            private["shed"][item] -= take
            inv[item] = inv.get(item, 0) + take
            return

        if op == "PLACE":
            if len(action) < 2:
                return
            item = action[1]
            if (item in ANIMALS and isinstance(tile, dict) and tile.get("kind") == ANIMALS[item]["structure"] and "animal" not in tile):
                if inv.get(item, 0) >= 1:
                    inv[item] -= 1
                    if inv[item] == 0:
                        del inv[item]
                    farm["tiles"][fy][fx] = {
                        "kind": ANIMALS[item]["structure"],
                        "animal": item,
                        "placed_day": day,
                        "yield_units": 0,
                        "consecutive_unfed": 0,
                        "fed_today": False,
                        "cared_today": False,
                        "fertilizer_available": False,
                        "pending_care_bonus": 0,
                    }
                return
            if _is_shed_adjacent((fx, fy), board_size):
                n = int(action[2]) if len(action) >= 3 else 1
                if n <= 0:
                    return
                take = min(n, inv.get(item, 0))
                if take <= 0:
                    return
                current = sum(private["shed"].values())
                room = max(0, shed_capacity - current)
                take = min(take, room)
                if take <= 0:
                    return
                inv[item] -= take
                if inv[item] == 0:
                    del inv[item]
                private["shed"][item] = private["shed"].get(item, 0) + take
            return

        if tile == "LOCKED":
            return

        if op == "PLANT":
            if len(action) < 2:
                return
            crop = action[1]
            if crop not in CROPS or tile is not None:
                return
            if private["seeds"].get(crop, 0) <= 0:
                return
            private["seeds"][crop] -= 1
            cd = CROPS[crop]
            farm["tiles"][fy][fx] = {
                "kind": "PLANT",
                "crop": crop,
                "planted_day": day,
                "watered_today": False,
                "consecutive_unwatered": 1,
                "yield_units": 0 if cd["ongoing"] else 1,
                "max_lifespan_step": (-1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day),
                "fertilized_until_day": -1,
            }
            return

        if op == "WATER":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                return
            if tile["watered_today"]:
                return
            tile["watered_today"] = True
            crop_data = CROPS[tile["crop"]]
            if not crop_data["ongoing"]:
                age_days = day - tile["planted_day"]
                window_start = (crop_data["max_yield_day"] + 1) // 2
                if window_start <= age_days <= crop_data["max_yield_day"]:
                    bonus = 2 if tile["fertilized_until_day"] >= day else 1
                    tile["yield_units"] = min(crop_data["max_yield"], tile["yield_units"] + bonus)
            return

        if op == "HARVEST":
            if not isinstance(tile, dict) or tile.get("yield_units", 0) <= 0:
                return
            if tile.get("kind") == "PLANT":
                crop_data = CROPS[tile["crop"]]
                if day - tile["planted_day"] < crop_data["first_yield_day"]:
                    return
                units = tile["yield_units"]
                tile["yield_units"] = 0
                inv[tile["crop"]] = inv.get(tile["crop"], 0) + units
                if not crop_data["ongoing"]:
                    farm["tiles"][fy][fx] = None
            elif "animal" in tile:
                units = tile["yield_units"]
                tile["yield_units"] = 0
                prod = ANIMALS[tile["animal"]]["product"]
                inv[prod] = inv.get(prod, 0) + units
            return

        if op == "FERTILIZE":
            if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
                return
            if inv.get("FERTILIZER", 0) <= 0:
                return
            inv["FERTILIZER"] -= 1
            if inv["FERTILIZER"] == 0:
                del inv["FERTILIZER"]
            tile["fertilized_until_day"] = max(tile.get("fertilized_until_day", -1), day + 2)
            return

        if op == "DIG":
            if tile is None or (isinstance(tile, dict) and "animal" in tile):
                return
            farm["tiles"][fy][fx] = None
            return

        if op == "BUILD_COOP":
            if tile is None:
                farm["tiles"][fy][fx] = {"kind": "COOP"}
            return

        if op == "BUILD_PASTURE":
            if tile is None:
                farm["tiles"][fy][fx] = {"kind": "PASTURE"}
            return

        if op == "FEED":
            if not (isinstance(tile, dict) and "animal" in tile) or tile["fed_today"]:
                return
            if inv.get("WHEAT", 0) <= 0:
                return
            inv["WHEAT"] -= 1
            if inv["WHEAT"] == 0:
                del inv["WHEAT"]
            tile["fed_today"] = True
            return

        if op == "COLLECT_FERTILIZER":
            if isinstance(tile, dict) and "animal" in tile and tile["fertilizer_available"]:
                tile["fertilizer_available"] = False
                inv["FERTILIZER"] = inv.get("FERTILIZER", 0) + 1
            return

        if op == "CARE":
            if isinstance(tile, dict) and "animal" in tile and not tile["cared_today"]:
                tile["cared_today"] = True
            return

    def _process_market(self, actions):
        max_orders = self.config["maxMarketOrdersPerTurn"]
        hire_mult = self.config["farmHandCostMult"]
        shed_capacity = self.config["shedCapacity"]
        board_size = self.config["boardSize"]

        queues = []
        for i in range(2):
            act = actions[i] if i < len(actions) and isinstance(actions[i], dict) else {}
            m = act.get("market", []) or []
            queues.append(list(m)[:max_orders])

        max_len = max((len(q) for q in queues), default=0)
        for i in range(max_len):
            order_states = []
            for player_id in range(2):
                q = queues[player_id]
                ostate = None
                if i < len(q):
                    raw = q[i]
                    if isinstance(raw, list) and raw:
                        top = raw[0]
                        if top in ("HIRE", "BUY_LAND"):
                            ostate = {"type": top}
                        elif top in ("BUY_SEED", "BUY_PRODUCT", "BUY_ANIMAL", "SELL") and len(raw) >= 3:
                            try:
                                count = int(raw[2])
                                if count > 0:
                                    ostate = {"type": top, "item": raw[1], "remaining": count}
                            except (ValueError, TypeError):
                                pass
                order_states.append(ostate)

            for pid in range(2):
                ost = order_states[pid]
                if ost is None:
                    continue
                if ost["type"] == "HIRE":
                    cost = hire_mult * _fib(self.farms[pid]["hires_today"])
                    if self.farms[pid]["money"] >= cost:
                        self.farms[pid]["money"] -= cost
                        self.farms[pid]["hires_today"] += 1
                        self.farms[pid]["hands"].append(_spawn_hand(self.farms[pid], board_size))
                        self.privates[pid]["inventories"].append({})
                    order_states[pid] = None
                elif ost["type"] == "BUY_LAND":
                    n_unlocked = len(self.farms[pid]["unlocked_quadrants"]) - 1
                    if n_unlocked < len(LAND_ORDER):
                        cost = LAND_PRICES[n_unlocked]
                        if self.farms[pid]["money"] >= cost:
                            self.farms[pid]["money"] -= cost
                            q_name = LAND_ORDER[n_unlocked]
                            self.farms[pid]["unlocked_quadrants"].append(q_name)
                            for y in range(board_size):
                                for x in range(board_size):
                                    if _quadrant_of(x, y, board_size) == q_name and self.farms[pid]["tiles"][y][x] == "LOCKED":
                                        self.farms[pid]["tiles"][y][x] = None
                    order_states[pid] = None

            while True:
                quoted = [None, None]
                for pid in range(2):
                    ost = order_states[pid]
                    if ost is None or ost.get("remaining", 0) <= 0:
                        continue
                    op = ost["type"]
                    item = ost["item"]
                    if op == "SELL" and item in PRODUCTS:
                        quoted[pid] = ("SELL", item, market_price(item, self.market["inventory"][item]), ost)
                    elif op == "BUY_PRODUCT" and item in ("WHEAT", "FERTILIZER"):
                        quoted[pid] = ("BUY_PRODUCT", item, market_price(item, self.market["inventory"][item] - 1), ost)
                    elif op == "BUY_SEED" and item in CROPS:
                        quoted[pid] = ("BUY_SEED", item, CROPS[item]["seed"], ost)
                    elif op == "BUY_ANIMAL" and item in ANIMALS:
                        quoted[pid] = ("BUY_ANIMAL", item, ANIMALS[item]["cost"], ost)
                    else:
                        order_states[pid] = None

                if all(q is None for q in quoted):
                    break

                committed_any = False
                for pid in range(2):
                    q = quoted[pid]
                    if q is None:
                        continue
                    op, item, price, ost = q
                    farm = self.farms[pid]
                    private = self.privates[pid]
                    ok = False
                    if op == "SELL":
                        if private["shed"].get(item, 0) > 0:
                            private["shed"][item] -= 1
                            farm["money"] += price
                            if price > 1:
                                self.market["inventory"][item] += 1
                            ok = True
                    elif op == "BUY_PRODUCT":
                        if farm["money"] >= price and sum(private["shed"].values()) < shed_capacity:
                            farm["money"] -= price
                            private["shed"][item] = private["shed"].get(item, 0) + 1
                            self.market["inventory"][item] -= 1
                            ok = True
                    elif op == "BUY_SEED":
                        if farm["money"] >= price:
                            farm["money"] -= price
                            private["seeds"][item] = private["seeds"].get(item, 0) + 1
                            ok = True
                    elif op == "BUY_ANIMAL":
                        if farm["money"] >= price and sum(private["shed"].values()) < shed_capacity:
                            farm["money"] -= price
                            private["shed"][item] = private["shed"].get(item, 0) + 1
                            ok = True

                    if ok:
                        ost["remaining"] -= 1
                        committed_any = True
                    else:
                        order_states[pid] = None

                if not committed_any:
                    break

            _refresh_prices(self.market)

    def _town_consume(self):
        shop_interval = self.config["townShopSellInterval"]
        center_interval = self.config["townCenterSellInterval"]

        if self.step_count % shop_interval == 0:
            for shop_name in self.town.get("unlocked_shops", []):
                products = SHOPS.get(shop_name, [])
                mult = 2 if len(products) == 1 else 1
                for item in products:
                    self.market["inventory"][item] -= mult

        if self.step_count % center_interval == 0:
            for item in TOWN_CENTER_PRODUCTS:
                self.market["inventory"][item] -= 1

        _refresh_prices(self.market)

    def _decay_plants(self, farm):
        board_size = len(farm["tiles"])
        for y in range(board_size):
            for x in range(board_size):
                tile = farm["tiles"][y][x]
                if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                    continue
                mls = tile["max_lifespan_step"]
                if mls < 0 or self.step_count < mls:
                    continue
                if (self.step_count - mls) % 2 != 0:
                    continue
                tile["yield_units"] -= 1
                if tile["yield_units"] <= 0:
                    farm["tiles"][y][x] = {"kind": "WEED"}

    def _end_of_day(self):
        board_size = self.config["boardSize"]
        shed_cap = self.config["shedCapacity"]
        weed_chance = self.config["weedSpawnChance"]
        turns_per_day = self.config["turnsPerDay"]
        day = self.day

        day_rng = random.Random((self.seed * 1_000_003) ^ day)

        for pid in range(2):
            farm = self.farms[pid]
            private = self.privates[pid]

            # 1. Plants refresh
            next_day = day + 1
            for y in range(board_size):
                for x in range(board_size):
                    tile = farm["tiles"][y][x]
                    if not isinstance(tile, dict) or tile.get("kind") != "PLANT":
                        continue
                    was_watered = tile["watered_today"]
                    if was_watered:
                        tile["consecutive_unwatered"] = 0
                    else:
                        tile["consecutive_unwatered"] += 1
                    tile["watered_today"] = False
                    if tile["consecutive_unwatered"] >= 2:
                        farm["tiles"][y][x] = {"kind": "WEED"}
                        continue
                    cd = CROPS[tile["crop"]]
                    if not cd["ongoing"]:
                        continue
                    days_since_first = next_day - tile["planted_day"] - cd["first_yield_day"]
                    if days_since_first < 0 or days_since_first % cd["interval"] != 0:
                        continue
                    production_count = days_since_first // cd["interval"] + 1
                    if production_count > cd["max_yield"]:
                        continue
                    fertilized = was_watered and tile.get("fertilized_until_day", -1) >= day
                    tile["yield_units"] = min(cd["max_yield"], tile["yield_units"] + (2 if fertilized else 1))
                    if production_count == cd["max_yield"]:
                        tile["max_lifespan_step"] = (next_day + 1) * turns_per_day

            # 2. Animals refresh
            for y in range(board_size):
                for x in range(board_size):
                    tile = farm["tiles"][y][x]
                    if not (isinstance(tile, dict) and "animal" in tile):
                        continue
                    if tile["fed_today"]:
                        tile["consecutive_unfed"] = 0
                    else:
                        tile["consecutive_unfed"] += 1
                    if tile["consecutive_unfed"] >= 2:
                        farm["tiles"][y][x] = {"kind": ANIMALS[tile["animal"]]["structure"]}
                        continue
                    a = ANIMALS[tile["animal"]]
                    days_since_first = next_day - tile["placed_day"] - a["first_yield_day"]
                    if days_since_first >= 0 and days_since_first % a["interval"] == 0:
                        bonus = tile.pop("pending_care_bonus", 0) if tile["fed_today"] else 0
                        tile["yield_units"] = min(a["max_held"], tile["yield_units"] + 1 + bonus)
                        tile["pending_care_bonus"] = 0
                    if tile["cared_today"] and tile["fed_today"]:
                        tile["pending_care_bonus"] = tile.get("pending_care_bonus", 0) + 1
                    tile["fertilizer_available"] = True
                    tile["fed_today"] = False
                    tile["cared_today"] = False

            # 3. Weeds spawn
            for y in range(board_size):
                for x in range(board_size):
                    if farm["tiles"][y][x] is None and day_rng.random() < weed_chance:
                        farm["tiles"][y][x] = {"kind": "WEED"}

            # 4. Inventories drop to shed
            shed = private["shed"]
            for inv in private["inventories"]:
                for item, n in list(inv.items()):
                    if n <= 0:
                        del inv[item]
                        continue
                    room = max(0, shed_cap - sum(shed.values()))
                    take = min(n, room)
                    if take > 0:
                        shed[item] = shed.get(item, 0) + take
                    del inv[item]

            # 5. Reset units
            farm["farmer"] = list(_default_spawn(board_size))
            farm["hands"] = []
            farm["hires_today"] = 0
            private["inventories"] = [{}]

        # 6. Town unlocks
        next_day = day + 1
        if next_day > 0 and next_day % self.config["townShopUnlockInterval"] == 0:
            if len(self.town["unlocked_shops"]) < MAX_SHOP_INSTANCES:
                self.town["unlocked_shops"].append(day_rng.choice(sorted(SHOPS)))

    def run(self, agent_funcs):
        """Run game until done. Returns final scores: [score_p0, score_p1]."""
        while not self.done:
            obs0 = self.get_observation(0)
            obs1 = self.get_observation(1)
            act0 = agent_funcs[0](obs0)
            act1 = agent_funcs[1](obs1)
            self.step([act0, act1])
        return [self.farms[0]["money"], self.farms[1]["money"]]
