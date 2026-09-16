with open("06_kaggriculture/main.py", "r", encoding="utf-8") as f:
    code = f.read()

# Make shed pickup happen before current tile animal actions if holding 0 wheat
old_tile_check = """        # --- 1. CURRENT TILE ACTIONS ---
        if isinstance(tile, dict):
            # Animal tile
            if "animal" in tile:
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"], (ux, uy)
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not tile.get("cared_today", False):
                    return ["CARE"], (ux, uy)"""

new_tile_check = """        # --- 1. PRIORITY SHED INTERACTION (Top Priority at Shed) ---
        if (ux, uy) in SHED_ADJACENT:
            # If holding goods, drop them off
            if inv.get("FERTILIZER", 0) >= 3 or inv.get("MILK", 0) >= 3 or inv.get("WOOL", 0) >= 2:
                return ["DROP"], (ux, uy)

            # Husbandry: pick up wheat FIRST if holding 0 wheat and shed has wheat
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

            # Husbandry: pick up animal if shed has one and spot available
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

        # --- 2. CURRENT TILE ACTIONS ---
        if isinstance(tile, dict):
            # Animal tile
            if "animal" in tile:
                if not tile.get("fed_today", False) and inv.get("WHEAT", 0) > 0:
                    return ["FEED"], (ux, uy)
                if not tile.get("cared_today", False):
                    return ["CARE"], (ux, uy)
                if tile.get("yield_units", 0) > 0:
                    return ["HARVEST"], (ux, uy)
                if tile.get("fertilizer_available", False):
                    return ["COLLECT_FERTILIZER"], (ux, uy)"""

assert old_tile_check in code, "old_tile_check not found!"
code = code.replace(old_tile_check, new_tile_check)

with open("06_kaggriculture/main.py", "w", encoding="utf-8") as f:
    f.write(code)

print("Successfully moved shed pickup to top priority!")
