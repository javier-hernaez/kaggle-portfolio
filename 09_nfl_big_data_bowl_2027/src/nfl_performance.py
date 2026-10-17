"""
NFL Performance Target Aggregator V2 for NFL Big Data Bowl 2027.
Processes 314,000+ regular season plays and multi-season records across all
five major position groups: WR, TE, DB, DL/EDGE, and OL, compiling the
Career Success Index (CSI) and specialized on-field metrics.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


def load_play_data() -> pd.DataFrame:
    """Load player_play.csv with key performance columns."""
    usecols = [
        "game_id", "play_id", "nfl_id", "lined_up_position",
        "expected_points_added", "route_ran",
        "separation_at_pass_forward", "target", "rec_yards",
        "yards_after_catch", "expected_yards_after_catch",
        "pressure_allowed", "sack_allowed", "quick_pressure",
        "player_get_off", "sack", "tackle_for_loss", "tackle"
    ]
    df = pd.read_csv(DATA_DIR / "player_play.csv", usecols=usecols)
    return df


def aggregate_pass_catcher_metrics(df_plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate receiver separation, EPA per target, and YAC over expectation."""
    route_plays = df_plays[df_plays["route_ran"].notnull() | (df_plays["target"] == 1)].copy()
    route_plays["yac_oe"] = route_plays["yards_after_catch"] - route_plays["expected_yards_after_catch"]

    agg = route_plays.groupby("nfl_id").agg({
        "separation_at_pass_forward": ["mean", "median", "count"],
        "target": "sum",
        "rec_yards": "sum",
        "expected_points_added": "mean",
        "yac_oe": "mean"
    })
    agg.columns = [
        "wr_mean_separation", "wr_median_separation", "wr_target_separation_samples",
        "wr_total_targets", "wr_total_rec_yards",
        "wr_mean_target_epa", "wr_mean_yac_oe"
    ]
    return agg.reset_index()


def aggregate_pass_rush_metrics(df_plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate edge/DL get-off, quick pressure rate, sacks, and TFLs."""
    for col in ["quick_pressure", "sack", "tackle_for_loss"]:
        df_plays[col] = pd.to_numeric(df_plays[col], errors="coerce").fillna(0)

    agg = df_plays.groupby("nfl_id").agg({
        "player_get_off": ["mean", "min", "count"],
        "quick_pressure": "sum",
        "sack": "sum",
        "tackle_for_loss": "sum"
    })
    agg.columns = [
        "dl_mean_get_off", "dl_min_get_off", "dl_snap_count",
        "dl_quick_pressures", "dl_total_sacks", "dl_total_tfl"
    ]
    df_res = agg.reset_index()
    df_res["dl_quick_pressure_rate"] = df_res["dl_quick_pressures"] / df_res["dl_snap_count"].clip(lower=1)
    return df_res


def aggregate_db_metrics(df_plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate defensive back closure and defensive play impact."""
    db_plays = df_plays[df_plays["lined_up_position"].isin(["CB", "SS", "FS", "DB"])].copy()
    if db_plays.empty:
        return pd.DataFrame()

    agg = db_plays.groupby("nfl_id").agg({
        "separation_at_pass_forward": "mean",
        "expected_points_added": "mean",
        "tackle": "sum",
        "play_id": "count"
    })
    agg.columns = [
        "db_allowed_separation", "db_mean_epa_allowed",
        "db_total_tackles", "db_total_coverage_snaps"
    ]
    return agg.reset_index()


def aggregate_ol_metrics(df_plays: pd.DataFrame) -> pd.DataFrame:
    """Aggregate offensive line pass-blocking efficiency."""
    ol_plays = df_plays[df_plays["pressure_allowed"].notnull()].copy()
    ol_plays["pressure_allowed"] = pd.to_numeric(ol_plays["pressure_allowed"], errors="coerce").fillna(0)
    ol_plays["sack_allowed"] = pd.to_numeric(ol_plays["sack_allowed"], errors="coerce").fillna(0)

    agg = ol_plays.groupby("nfl_id").agg({
        "pressure_allowed": ["mean", "sum", "count"],
        "sack_allowed": "sum"
    })
    agg.columns = [
        "ol_pressure_allowed_rate", "ol_total_pressures_allowed",
        "ol_pass_block_snaps", "ol_sacks_allowed"
    ]
    df_res = agg.reset_index()
    # Pass Blocking Efficiency formula: 100 - (pressures + 2*sacks) / snaps * 100
    df_res["ol_pass_blocking_efficiency"] = 100.0 - (
        (df_res["ol_total_pressures_allowed"] + 2.0 * df_res["ol_sacks_allowed"]) /
        df_res["ol_pass_block_snaps"].clip(lower=1) * 100.0
    ).clip(lower=0, upper=100)
    return df_res


def load_career_metrics() -> pd.DataFrame:
    """Load player_career_successes.csv and compute standardized career metrics + CSI."""
    career = pd.read_csv(DATA_DIR / "player_career_successes.csv")
    career["career_total_snaps"] = career["career_offensive_snaps"] + career["career_defensive_snaps"]
    career["starter_rate"] = career["career_games_started"] / career["career_games_active"].clip(lower=1)
    career["is_starter"] = (career["career_games_started"] >= 8).astype(int)
    career["all_pro_or_pro_bowl"] = (
        (career["ap_all_pro_1st_team"] > 0) |
        (career["ap_all_pro_2nd_team"] > 0) |
        (career["pro_bowl_original_ballot"] > 0)
    ).astype(int)

    # Career Success Index (CSI): 0 - 100 scale composite
    snap_pct = career["career_total_snaps"].rank(pct=True) * 40.0
    start_pct = career["career_games_started"].rank(pct=True) * 40.0
    honors_bonus = career["all_pro_or_pro_bowl"] * 20.0
    career["career_success_index"] = (snap_pct + start_pct + honors_bonus).clip(0, 100)
    return career


def build_performance_targets() -> pd.DataFrame:
    """Build unified multi-target ground truth dataset for all prospects."""
    print("=== Aggregating NFL Performance Ground Truth Targets V2 ===")
    players = pd.read_csv(DATA_DIR / "players.csv")
    print(f"[*] Base prospects: {len(players)}")

    print("[*] Processing regular-season plays (314,000+ rows)...")
    df_plays = load_play_data()

    print("[*] Aggregating pass-catcher performance (Separation, EPA, xYAC)...")
    wr_metrics = aggregate_pass_catcher_metrics(df_plays)

    print("[*] Aggregating pass-rush performance (Get-off, Quick Pressures, Sacks)...")
    dl_metrics = aggregate_pass_rush_metrics(df_plays)

    print("[*] Aggregating defensive back performance (Closure, EPA allowed)...")
    db_metrics = aggregate_db_metrics(df_plays)

    print("[*] Aggregating offensive line pass-blocking efficiency...")
    ol_metrics = aggregate_ol_metrics(df_plays)

    print("[*] Loading career progression, starter rates, and CSI...")
    career_metrics = load_career_metrics()

    # Merge everything onto players
    df_targets = players[["nfl_id", "display_name", "draft_year", "nfl_position", "draft_overall_pick"]].copy()
    df_targets = df_targets.merge(career_metrics, on="nfl_id", how="left")
    df_targets = df_targets.merge(wr_metrics, on="nfl_id", how="left")
    df_targets = df_targets.merge(dl_metrics, on="nfl_id", how="left")
    if not db_metrics.empty:
        df_targets = df_targets.merge(db_metrics, on="nfl_id", how="left")
    df_targets = df_targets.merge(ol_metrics, on="nfl_id", how="left")

    out_path = DATA_DIR / "nfl_performance_targets.csv"
    df_targets.to_csv(out_path, index=False)
    print(f"[+] Compiled {df_targets.shape[1]} ground truth target metrics for {len(df_targets)} prospects.")
    print(f"[+] Saved to {out_path}")
    return df_targets


if __name__ == "__main__":
    build_performance_targets()
