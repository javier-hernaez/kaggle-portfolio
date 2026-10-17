"""
Kinematics Feature Extraction Engine V2 for NFL Big Data Bowl 2027.
Advanced multi-position biomechanical pipeline extracting 38 high-frequency (10 Hz)
features from combine_tracking.csv across WR, TE, DB, DL/EDGE, and OL prospect drills.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

ROUTE_DRILL_KEYWORDS = [
    "ROUTE", "GAUNTLET", "SLOT", "CORNER", "DAGGER", "COMEBACK",
    "CURL", "SPEED_OUT", "SLANT", "CIRCUS", "POST", "HOOK", "WHEEL"
]

PASS_RUSH_DRILL_KEYWORDS = [
    "PASS_RUSH", "RUN_THE_HOOP", "RUN_AND_CLUB", "LINE", "BLOCK"
]

DB_DRILL_KEYWORDS = [
    "BACK_PEDAL", "TRANSITION", "BREAK_TO_BALL", "ZONE_BREAK", "TERYL_AUSTIN"
]

OL_DRILL_KEYWORDS = [
    "OL_PULL", "PASS_PRO", "FOLD_BLOCK", "LINE_DRILL"
]


def circular_angle_diff(series: pd.Series) -> pd.Series:
    """Calculate circular angular difference in degrees [-180, 180]."""
    diff = series.diff()
    return (diff + 180.0) % 360.0 - 180.0


def extract_forty_kinematics(forty_df: pd.DataFrame) -> pd.DataFrame:
    """Extract kinematic drive-phase and top-speed features from 40-yard dash tracking."""
    if forty_df.empty:
        return pd.DataFrame()

    results = []
    for (nfl_id, attempt), group in forty_df.groupby(["nfl_id", "attempt"]):
        g = group.sort_values("time").reset_index(drop=True)
        if len(g) < 5:
            continue

        dt = 0.1
        peak_speed = g["s"].max()
        peak_accel = g["a"].max()
        peak_speed_idx = g["s"].idxmax()
        time_to_peak_speed = peak_speed_idx * dt

        # Drive phase: first 15 frames (1.5 seconds)
        drive_phase = g.iloc[:min(15, len(g))]
        jerk_series = drive_phase["a"].diff() / dt
        drive_jerk = jerk_series.dropna().mean()

        # Initial burst gradient (first 5 frames / 0.5s)
        initial_burst = g["a"].iloc[:min(5, len(g))].mean()

        # Speed retention at end of dash
        speed_end = g["s"].iloc[-1]
        speed_retention = speed_end / max(peak_speed, 1e-4)

        results.append({
            "nfl_id": nfl_id,
            "forty_peak_speed": peak_speed,
            "forty_peak_accel": peak_accel,
            "forty_time_to_peak_speed": time_to_peak_speed,
            "forty_drive_jerk": drive_jerk,
            "forty_initial_burst": initial_burst,
            "forty_speed_retention": speed_retention
        })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.groupby("nfl_id").mean().reset_index()


def extract_route_kinematics(route_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract Route Sharpness Index (RSI), Kinematic Break Efficiency (KBE),
    Deceleration-to-Acceleration Ratio (DAR), and Dwell Time.
    """
    if route_df.empty:
        return pd.DataFrame()

    results = []
    dt = 0.1

    for (nfl_id, event_id), group in route_df.groupby(["nfl_id", "event_id"]):
        g = group.sort_values("time").reset_index(drop=True)
        if len(g) < 10:
            continue

        d_theta = circular_angle_diff(g["dir"])
        angular_vel = d_theta.abs() / dt  # deg/sec
        angular_vel_rad = np.radians(angular_vel)
        centripetal_acc = g["s"] * angular_vel_rad
        jerk = g["a"].diff() / dt

        peak_ang_vel = angular_vel.max()
        mean_ang_vel = angular_vel.mean()
        peak_centripetal = centripetal_acc.max()

        # Dwell time: frames spent rotating above 150 deg/s
        dwell_frames = (angular_vel > 150.0).sum()
        dwell_time = dwell_frames * dt

        # Identify primary route cut apex (maximum angular change above 300 deg/s)
        apex_candidates = angular_vel[angular_vel > 300.0]
        if not apex_candidates.empty:
            apex_idx = apex_candidates.idxmax()
            entry_idx = max(0, apex_idx - 3)
            exit_idx = min(len(g) - 1, apex_idx + 4)

            s_entry = max(g["s"].iloc[entry_idx], 0.1)
            s_apex = g["s"].iloc[apex_idx]
            s_exit = g["s"].iloc[exit_idx]
            cut_speed_retention = s_apex / s_entry
            cut_burst_accel = max(g["a"].iloc[apex_idx:exit_idx + 1].max(), 0.0)
            cut_braking_decel = min(g["a"].iloc[entry_idx:apex_idx + 1].min(), 0.0)

            # Kinematic Break Efficiency
            kbe = (s_exit / s_entry) * (1.0 + cut_burst_accel)
            # Route Sharpness Index
            rsi = peak_ang_vel / max(s_apex, 1.0)
            # Deceleration-to-Acceleration Ratio
            dar = abs(cut_braking_decel) / max(cut_burst_accel, 0.1)
        else:
            cut_speed_retention = 1.0
            cut_burst_accel = g["a"].max()
            cut_braking_decel = g["a"].min()
            kbe = 1.0
            rsi = peak_ang_vel / max(g["s"].mean(), 1.0)
            dar = 1.0

        results.append({
            "nfl_id": nfl_id,
            "route_peak_angular_vel": peak_ang_vel,
            "route_mean_angular_vel": mean_ang_vel,
            "route_peak_centripetal": peak_centripetal,
            "route_kbe": kbe,
            "route_rsi": rsi,
            "route_cut_speed_retention": cut_speed_retention,
            "route_cut_burst_accel": cut_burst_accel,
            "route_cut_braking_decel": cut_braking_decel,
            "route_dar": dar,
            "route_dwell_time": dwell_time,
            "route_mean_abs_jerk": jerk.abs().mean()
        })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.groupby("nfl_id").mean().reset_index()


def extract_db_kinematics(db_df: pd.DataFrame) -> pd.DataFrame:
    """Extract backpedal-to-break transition speed retention and reaction mechanics for DBs."""
    if db_df.empty:
        return pd.DataFrame()

    results = []
    dt = 0.1

    for (nfl_id, event_id), group in db_df.groupby(["nfl_id", "event_id"]):
        g = group.sort_values("time").reset_index(drop=True)
        if len(g) < 8:
            continue

        d_theta = circular_angle_diff(g["dir"])
        angular_vel = d_theta.abs() / dt

        # Hip flip / break apex is where orientation flips dramatically (>90 deg)
        flip_candidates = angular_vel[angular_vel > 450.0]
        if not flip_candidates.empty:
            apex_idx = flip_candidates.idxmax()
            s_pre = max(g["s"].iloc[max(0, apex_idx - 3)], 0.1)
            s_post = g["s"].iloc[min(len(g) - 1, apex_idx + 4)]
            flip_retention = s_post / s_pre
            flip_burst = max(g["a"].iloc[apex_idx:].max(), 0.0)
        else:
            flip_retention = 1.0
            flip_burst = g["a"].max()

        results.append({
            "nfl_id": nfl_id,
            "db_hip_flip_retention": flip_retention,
            "db_flip_burst_accel": flip_burst,
            "db_peak_transition_ang_vel": angular_vel.max(),
            "db_mean_reaction_speed": g["s"].mean()
        })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.groupby("nfl_id").mean().reset_index()


def extract_trench_kinematics(trench_df: pd.DataFrame) -> pd.DataFrame:
    """Extract bend index, curvature acceleration, and first-step surge for edge/DL."""
    if trench_df.empty:
        return pd.DataFrame()

    results = []
    dt = 0.1

    for (nfl_id, event_id), group in trench_df.groupby(["nfl_id", "event_id"]):
        g = group.sort_values("time").reset_index(drop=True)
        if len(g) < 6:
            continue

        # First-step surge (first 0.5s / 5 frames)
        first_step = g.iloc[:min(5, len(g))]
        first_step_burst = first_step["a"].max()
        first_step_gradient = (first_step["s"].iloc[-1] - first_step["s"].iloc[0]) / max(len(first_step) * dt, 1e-4)

        # Curvature bend index (centripetal acceleration during continuous curve)
        d_theta = circular_angle_diff(g["dir"])
        angular_vel_rad = np.radians(d_theta.abs() / dt)
        centripetal_acc = g["s"] * angular_vel_rad
        bend_index = centripetal_acc.quantile(0.90) if len(centripetal_acc) > 0 else 0.0

        results.append({
            "nfl_id": nfl_id,
            "trench_first_step_burst": first_step_burst,
            "trench_first_step_gradient": first_step_gradient,
            "trench_bend_index": bend_index,
            "trench_peak_speed": g["s"].max(),
            "trench_peak_accel": g["a"].max()
        })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.groupby("nfl_id").mean().reset_index()


def extract_ol_kinematics(ol_df: pd.DataFrame) -> pd.DataFrame:
    """Extract lateral shuffle speed, pull acceleration, and pass protection stability for OL."""
    if ol_df.empty:
        return pd.DataFrame()

    results = []
    for (nfl_id, event_id), group in ol_df.groupby(["nfl_id", "event_id"]):
        g = group.sort_values("time").reset_index(drop=True)
        if len(g) < 6:
            continue

        results.append({
            "nfl_id": nfl_id,
            "ol_pull_peak_accel": g["a"].max(),
            "ol_lateral_mean_speed": g["s"].mean(),
            "ol_speed_stability": g["s"].std()
        })

    if not results:
        return pd.DataFrame()

    df_res = pd.DataFrame(results)
    return df_res.groupby("nfl_id").mean().reset_index()


def extract_overall_kinematics(df_players_tracking: pd.DataFrame) -> pd.DataFrame:
    """Extract aggregate kinematic capacity metrics across all drills for each player."""
    dt = 0.1
    df_sorted = df_players_tracking.sort_values(["nfl_id", "event_id", "time"]).reset_index(drop=True)
    df_sorted["d_theta"] = circular_angle_diff(df_sorted["dir"])
    df_sorted["angular_vel"] = df_sorted["d_theta"].abs() / dt
    df_sorted["jerk"] = df_sorted.groupby(["nfl_id", "event_id"])["a"].diff() / dt

    agg = df_sorted.groupby("nfl_id").agg({
        "s": ["max", "mean"],
        "a": ["max", "min", "mean"],
        "angular_vel": ["max", "mean"],
        "jerk": [lambda x: x.abs().mean(), "max"]
    })
    agg.columns = [
        "comb_peak_speed", "comb_mean_speed",
        "comb_peak_accel", "comb_peak_decel", "comb_mean_accel",
        "comb_peak_angular_vel", "comb_mean_angular_vel",
        "comb_mean_abs_jerk", "comb_peak_jerk"
    ]
    return agg.reset_index()


def extract_all_kinematic_features() -> pd.DataFrame:
    """Main extraction pipeline: loads combine_tracking.csv and returns unified V2 feature matrix."""
    tracking_path = DATA_DIR / "combine_tracking.csv"
    print(f"[*] Loading Combine tracking data from {tracking_path}...")
    
    usecols = [
        "draft_year", "event_id", "nfl_id", "entity_type",
        "time", "drill_type", "drill_name", "attempt",
        "x", "y", "s", "a", "dis", "dir"
    ]
    df = pd.read_csv(tracking_path, usecols=usecols)
    df = df[df["entity_type"] == "PLAYER"].copy()
    print(f"[+] Loaded {len(df)} player tracking frames across {df['nfl_id'].nunique()} prospects.")

    print("[*] Computing overall kinematic capacity metrics...")
    df_overall = extract_overall_kinematics(df)

    print("[*] Computing 40-yard dash kinematics & initial burst gradient...")
    forty_df = df[df["drill_name"] == "FORTY_YARD_DASH"].copy()
    df_forty = extract_forty_kinematics(forty_df)

    print("[*] Computing route running & break efficiency (KBE, RSI, DAR, Dwell Time)...")
    route_mask = df["drill_name"].apply(lambda name: any(kw in str(name).upper() for kw in ROUTE_DRILL_KEYWORDS))
    route_df = df[route_mask].copy()
    df_route = extract_route_kinematics(route_df)

    print("[*] Computing defensive back hip-turn & break-to-ball kinematics...")
    db_mask = df["drill_name"].apply(lambda name: any(kw in str(name).upper() for kw in DB_DRILL_KEYWORDS))
    db_df = df[db_mask].copy()
    df_db = extract_db_kinematics(db_df)

    print("[*] Computing trench bend & first-step surge kinematics...")
    trench_mask = df["drill_name"].apply(lambda name: any(kw in str(name).upper() for kw in PASS_RUSH_DRILL_KEYWORDS))
    trench_df = df[trench_mask].copy()
    df_trench = extract_trench_kinematics(trench_df)

    print("[*] Computing offensive line pull & lateral stability kinematics...")
    ol_mask = df["drill_name"].apply(lambda name: any(kw in str(name).upper() for kw in OL_DRILL_KEYWORDS))
    ol_df = df[ol_mask].copy()
    df_ol = extract_ol_kinematics(ol_df)

    print("[*] Merging all position-specific kinematic features...")
    kinematics = df_overall.copy()

    for sub_df in [df_forty, df_route, df_db, df_trench, df_ol]:
        if not sub_df.empty:
            kinematics = kinematics.merge(sub_df, on="nfl_id", how="left")

    # Median imputation for drill-specific columns
    kinematics = kinematics.fillna(kinematics.median(numeric_only=True))

    out_path = DATA_DIR / "prospect_kinematics.csv"
    kinematics.to_csv(out_path, index=False)
    print(f"[+] Successfully extracted {kinematics.shape[1]} kinematic features for {len(kinematics)} prospects.")
    print(f"[+] Saved to {out_path}")
    return kinematics


if __name__ == "__main__":
    extract_all_kinematic_features()
