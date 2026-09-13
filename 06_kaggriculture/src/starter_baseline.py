"""
Baseline agents for Kaggriculture.
Includes the official starter agent, random agent, and pass agent.
"""

import random

CROPS = {
    "WHEAT":      {"seed": 10, "first_yield_day": 2, "max_yield_day": 4, "interval": 0, "max_yield": 6, "ongoing": False},
    "CARROT":     {"seed": 20, "first_yield_day": 2, "max_yield_day": 3, "interval": 0, "max_yield": 4, "ongoing": False},
    "TOMATO":     {"seed": 50, "first_yield_day": 8, "max_yield_day": 8, "interval": 1, "max_yield": 4, "ongoing": True},
    "STRAWBERRY": {"seed": 100, "first_yield_day": 10, "max_yield_day": 10, "interval": 2, "max_yield": 4, "ongoing": True},
    "MELON":      {"seed": 80, "first_yield_day": 10, "max_yield_day": 12, "interval": 0, "max_yield": 6, "ongoing": False},
}


def pass_agent(obs):
    return {"farmer": ["PASS"], "hands": [], "market": []}


def random_agent(obs):
    rng = random.Random()
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    farm = farms[player] if farms and player < len(farms) else None
    if farm is None:
        return {"farmer": ["PASS"], "hands": [], "market": []}

    farmer_ops = ["NORTH", "SOUTH", "EAST", "WEST", "WATER", "HARVEST", "PASS"]
    market = []
    seeds = private.get("seeds", {})

    affordable = [c for c in CROPS if CROPS[c]["seed"] <= farm["money"]]
    if affordable and rng.random() < 0.1:
        market.append(["BUY_SEED", rng.choice(affordable), 1])

    available_seeds = [c for c, n in seeds.items() if n > 0]
    if available_seeds and rng.random() < 0.3:
        farmer = ["PLANT", rng.choice(available_seeds)]
    else:
        farmer = [rng.choice(farmer_ops)]

    hands_actions = [[rng.choice(farmer_ops)] for _ in farm.get("hands", [])]
    return {"farmer": farmer, "hands": hands_actions, "market": market}


def starter_agent(obs):
    """
    Official Kaggle Starter Agent:
    Carrot loop: buy carrot seed, plant on current tile, water, harvest at max_yield_day.
    """
    farms = obs.get("farms", [])
    player = obs.get("player", 0)
    private = obs.get("private", {}) or {}
    if not farms or player >= len(farms):
        return {"farmer": ["PASS"], "hands": [], "market": []}
    farm = farms[player]
    fx, fy = farm["farmer"]
    tile = farm["tiles"][fy][fx]
    day = obs.get("day", 0)
    seeds = private.get("seeds", {})
    shed = private.get("shed", {})

    market = []
    if shed.get("CARROT", 0) > 0:
        market.append(["SELL", "CARROT", shed["CARROT"]])
    if seeds.get("CARROT", 0) == 0 and farm["money"] >= CROPS["CARROT"]["seed"]:
        market.append(["BUY_SEED", "CARROT", 1])

    farmer = ["PASS"]
    if tile is None and seeds.get("CARROT", 0) > 0:
        farmer = ["PLANT", "CARROT"]
    elif isinstance(tile, dict) and tile.get("kind") == "PLANT" and tile["crop"] == "CARROT":
        age = day - tile["planted_day"]
        if age >= CROPS["CARROT"]["max_yield_day"]:
            farmer = ["HARVEST"]
        elif not tile["watered_today"]:
            farmer = ["WATER"]
    return {"farmer": farmer, "hands": farmer_ops if False else [], "market": market}
