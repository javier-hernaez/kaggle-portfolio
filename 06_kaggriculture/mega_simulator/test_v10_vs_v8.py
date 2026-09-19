import sys
sys.path.insert(0, '06_kaggriculture')

from mega_simulator.agent_genome import Genome
from mega_simulator.arena import Arena
import json

# Load v8 (Gen4_Mutant2) from Hall of Fame
with open("06_kaggriculture/mega_simulator/hall_of_fame.json", "r") as f:
    hof = json.load(f)

v8_dict = None
for entry in hof:
    if entry["genome"]["name"] == "Gen4_Mutant2":
        v8_dict = entry["genome"]
        break

if not v8_dict:
    # Fallback to Gen4_Mutant2 values
    v8_dict = {
        "name": "Gen4_Mutant2",
        "land_ne_min_day": 5, "land_ne_min_money": 1300,
        "land_sw_min_day": 11, "land_sw_min_money": 2500,
        "buy_se_land": False, "land_se_min_day": 16, "land_se_min_money": 4500,
        "target_cows": 10, "target_sheep": 5, "animal_end_day": 12,
        "cow_to_sheep_ratio": 2.5, "wheat_reserve_mult": 1,
        "strawberry_start_day": 5, "strawberry_end_day": 13,
        "target_strawberries": 45, "strawberry_seed_batch": 10,
        "labor_early": 7, "labor_mid": 7, "labor_mid_day": 6,
        "labor_late": 10, "labor_late_day": 10,
        "price_throttle_ratio": 0.65, "price_burst_ratio": 1.15,
        "batch_milk": 6, "batch_wool": 4, "batch_strawberry": 8,
        "batch_fertilizer": 12, "batch_melon": 6,
        "shop_icecream_straw_boost": 10, "shop_yarn_sheep_boost": 2,
        "skip_watering_if_rained": True
    }

genome_v8 = Genome.from_dict(v8_dict)

# Build v10 (Supreme Synthesis)
v10_dict = dict(v8_dict)
v10_dict["name"] = "Grandmaster_v10_Supreme"
v10_dict["target_strawberries"] = 48
v10_dict["target_cows"] = 10
v10_dict["target_sheep"] = 5
v10_dict["labor_early"] = 7
v10_dict["labor_mid"] = 8
v10_dict["labor_late"] = 12
v10_dict["wheat_reserve_mult"] = 1
v10_dict["land_ne_min_day"] = 5
v10_dict["land_ne_min_money"] = 1250
v10_dict["land_sw_min_day"] = 10
v10_dict["land_sw_min_money"] = 2400
v10_dict["price_burst_ratio"] = 1.15
v10_dict["price_throttle_ratio"] = 0.60
v10_dict["batch_fertilizer"] = 12
v10_dict["shop_icecream_straw_boost"] = 12
v10_dict["shop_yarn_sheep_boost"] = 2
v10_dict["skip_watering_if_rained"] = True

genome_v10 = Genome.from_dict(v10_dict)

if __name__ == '__main__':
    print("Running Mirror Duel: v10 (Supreme) vs v8 (Kaggle 573.3 Elo Champion)...")
    arena = Arena(max_workers=4)
    seeds = [42, 1039, 2036, 3033, 4030, 5027, 6024, 7021, 8018, 9015]
    res = arena.duel(genome_v10, genome_v8, seeds)

    print("\n=== DUEL RESULTS: v10 vs v8 (20 mirror games) ===")
    print(f"Total Games: {res['total_games']}")
    print(f"v10 Wins: {res['wins_a']} | v8 Wins: {res['wins_b']} | Ties: {res['ties']}")
    print(f"v10 Winrate: {res['winrate_a']*100:.1f}%")
    print(f"Average Score v10: ${res['avg_score_a']:,.0f}")
    print(f"Average Score v8:  ${res['avg_score_b']:,.0f}")
    print(f"Net Margin: ${res['net_margin']:+,.0f}")
    print(f"Is v10 Superior? {res['is_a_superior']}")
