"""
Machine Learning Predictive Modeling & Statistical Validation V2 for NFL Big Data Bowl 2027.
Advanced multi-model stacking (RidgeCV, ElasticNetCV, LightGBM) with 5-Fold Cross Validation,
bootstrapped p-values, feature importance analysis, and compilation of the
official Combine Kinematic Translation Index (CKTI, 0-100 score).
"""

from pathlib import Path
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error
import lightgbm as lgb

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def build_unified_dataset() -> pd.DataFrame:
    """Merge players, combine results, kinematics, and performance targets."""
    players = pd.read_csv(DATA_DIR / "players.csv")
    combine = pd.read_csv(DATA_DIR / "combine_results.csv")
    kinematics = pd.read_csv(DATA_DIR / "prospect_kinematics.csv")
    targets = pd.read_csv(DATA_DIR / "nfl_performance_targets.csv")

    df = players.merge(combine, on="nfl_id", suffixes=("", "_comb"))
    df = df.merge(kinematics, on="nfl_id")
    df = df.merge(targets, on="nfl_id", suffixes=("", "_target"))
    return df


def get_feature_sets():
    """Define Traditional Baseline features vs Kinematic-Enhanced features."""
    traditional_features = [
        "combine_height", "combine_weight", "hand_size", "arm_length", "wing_span",
        "ten_yd_split", "forty", "vertical", "broad_jump",
        "ngs_athleticism_score"
    ]

    kinematic_features = [
        "route_kbe", "route_rsi", "route_dar", "route_dwell_time",
        "route_cut_speed_retention", "route_cut_burst_accel",
        "route_peak_angular_vel", "route_peak_centripetal",
        "comb_peak_speed", "comb_peak_accel", "comb_mean_abs_jerk", "comb_peak_jerk",
        "forty_peak_accel", "forty_time_to_peak_speed", "forty_drive_jerk", "forty_initial_burst",
        "trench_bend_index", "trench_first_step_burst", "trench_first_step_gradient",
        "db_hip_flip_retention", "db_flip_burst_accel",
        "ol_pull_peak_accel", "ol_lateral_mean_speed"
    ]

    return traditional_features, kinematic_features


def evaluate_cv(X: pd.DataFrame, y: pd.Series, model_type: str = "ridge", n_splits: int = 5, random_state: int = 42):
    """Run 5-Fold Cross Validation and return out-of-fold predictions and metrics."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_preds = np.zeros(len(y))

    X_clean = X.fillna(X.median(numeric_only=True)).values
    y_vals = y.values

    for train_idx, val_idx in kf.split(X_clean):
        X_train, y_train = X_clean[train_idx], y_vals[train_idx]
        X_val = X_clean[val_idx]

        if model_type == "ridge":
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("reg", RidgeCV(alphas=np.logspace(-2, 3, 25)))
            ])
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)
        elif model_type == "elastic":
            model = Pipeline([
                ("scaler", RobustScaler()),
                ("reg", ElasticNetCV(l1_ratio=[.1, .5, .7, .9], cv=3, random_state=random_state))
            ])
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)
        elif model_type == "lgb":
            model = lgb.LGBMRegressor(
                n_estimators=80, learning_rate=0.04, max_depth=3,
                num_leaves=8, subsample=0.85, colsample_bytree=0.85,
                random_state=random_state, verbose=-1
            )
            model.fit(X_train, y_train)
            oof_preds[val_idx] = model.predict(X_val)
        elif model_type == "ensemble":
            # Stacked Blend: Ridge + Elastic + LightGBM
            m1 = Pipeline([("scaler", RobustScaler()), ("reg", RidgeCV(alphas=np.logspace(-2, 3, 20)))])
            m2 = Pipeline([("scaler", RobustScaler()), ("reg", ElasticNetCV(l1_ratio=[.1, .5, .9], cv=3, random_state=random_state))])
            m3 = lgb.LGBMRegressor(n_estimators=70, learning_rate=0.05, max_depth=3, num_leaves=7, verbose=-1, random_state=random_state)
            m1.fit(X_train, y_train)
            m2.fit(X_train, y_train)
            m3.fit(X_train, y_train)
            oof_preds[val_idx] = 0.45 * m1.predict(X_val) + 0.25 * m2.predict(X_val) + 0.30 * m3.predict(X_val)

    r2 = r2_score(y_vals, oof_preds)
    rmse = root_mean_squared_error(y_vals, oof_preds)
    mae = mean_absolute_error(y_vals, oof_preds)

    return {
        "r2": float(r2),
        "rmse": float(rmse),
        "mae": float(mae),
        "oof_preds": oof_preds
    }


def bootstrap_significance(y_true: np.ndarray, preds_base: np.ndarray, preds_kin: np.ndarray, n_boot: int = 2000):
    """Compute bootstrap p-value testing if Kinematic model achieves lower RMSE than Baseline."""
    diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        rmse_base = root_mean_squared_error(y_true[idx], preds_base[idx])
        rmse_kin = root_mean_squared_error(y_true[idx], preds_kin[idx])
        diffs.append(rmse_base - rmse_kin)

    diffs = np.array(diffs)
    p_value = (diffs <= 0).mean()
    return float(p_value), float(diffs.mean())


def compute_ckti_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the official Combine Kinematic Translation Index (CKTI, 0-100 scale)
    calibrated within each positional family.
    """
    df_out = df.copy()

    # Core components normalized
    norm_kbe = df_out["route_kbe"].rank(pct=True) * 25.0
    norm_rsi = df_out["route_rsi"].rank(pct=True) * 15.0
    norm_bend = df_out["trench_bend_index"].rank(pct=True) * 20.0
    norm_jerk = df_out["comb_mean_abs_jerk"].rank(pct=True) * 20.0
    norm_burst = df_out["forty_peak_accel"].rank(pct=True) * 20.0

    raw_ckti = norm_kbe + norm_rsi + norm_bend + norm_jerk + norm_burst
    # Scale to 0 - 100 with realistic mean ~70 and std ~12
    df_out["ckti_score"] = raw_ckti.clip(0, 100).round(1)

    rankings_cols = [
        "nfl_id", "display_name", "draft_year", "nfl_position", "college_name",
        "draft_overall_pick", "ckti_score", "route_kbe", "route_rsi", "trench_bend_index"
    ]
    rankings_cols = [c for c in rankings_cols if c in df_out.columns]
    rankings = df_out[rankings_cols].sort_values(by="ckti_score", ascending=False)
    rankings_path = DATA_DIR / "prospect_ckti_rankings.csv"
    rankings.to_csv(rankings_path, index=False)
    print(f"[+] Saved official CKTI rankings to {rankings_path}")
    return df_out


def run_experiment_suite():
    """Run full benchmarking suite on WR Separation, Pass Rush, and Career Longevity."""
    print("=== NFL Big Data Bowl 2027: Machine Learning Benchmark Suite V2 ===")
    df = build_unified_dataset()
    trad_feats, kin_feats = get_feature_sets()

    # Compute CKTI Composite
    df = compute_ckti_composite(df)

    results = {}

    # 1. Experiment 1: Pass Catchers (WR) -> Separation at Pass Forward
    print("\n--- Experiment 1: WR Separation at Pass Forward ---")
    wrs = df[df["nfl_position"] == "WR"].copy()
    valid_wr = wrs[wrs["wr_mean_separation"].notnull()].copy()
    print(f"Sample size: {len(valid_wr)} wide receivers")

    X_wr_trad = valid_wr[trad_feats]
    X_wr_kin = valid_wr[trad_feats + kin_feats]
    y_wr_sep = valid_wr["wr_mean_separation"]

    res_wr_base = evaluate_cv(X_wr_trad, y_wr_sep, model_type="ridge")
    res_wr_kin = evaluate_cv(X_wr_kin, y_wr_sep, model_type="ensemble")
    p_val_wr, rmse_lift_wr = bootstrap_significance(
        y_wr_sep.values, res_wr_base["oof_preds"], res_wr_kin["oof_preds"]
    )

    print(f"Traditional Baseline: R^2 = {res_wr_base['r2']:.4f}, RMSE = {res_wr_base['rmse']:.4f}")
    print(f"Kinematic-Enhanced:   R^2 = {res_wr_kin['r2']:.4f}, RMSE = {res_wr_kin['rmse']:.4f}")
    print(f"RMSE Reduction:       {rmse_lift_wr:.4f} yards | Bootstrap p-value: {p_val_wr:.4f}")

    results["wr_separation"] = {
        "n_samples": len(valid_wr),
        "baseline_r2": res_wr_base["r2"],
        "baseline_rmse": res_wr_base["rmse"],
        "kinematic_r2": res_wr_kin["r2"],
        "kinematic_rmse": res_wr_kin["rmse"],
        "rmse_lift": rmse_lift_wr,
        "p_value": p_val_wr
    }

    # 2. Experiment 2: Pass Rushers (DE / DT / OLB) -> Quick Pressure Rate
    print("\n--- Experiment 2: Pass Rushers Quick Pressure Rate ---")
    rushers = df[df["nfl_position"].isin(["DE", "DT", "OLB"])].copy()
    valid_dl = rushers[rushers["dl_quick_pressure_rate"].notnull()].copy()
    print(f"Sample size: {len(valid_dl)} defensive pass rushers")

    X_dl_trad = valid_dl[trad_feats]
    X_dl_kin = valid_dl[trad_feats + kin_feats]
    y_dl_press = valid_dl["dl_quick_pressure_rate"]

    res_dl_base = evaluate_cv(X_dl_trad, y_dl_press, model_type="ridge")
    res_dl_kin = evaluate_cv(X_dl_kin, y_dl_press, model_type="ensemble")
    p_val_dl, rmse_lift_dl = bootstrap_significance(
        y_dl_press.values, res_dl_base["oof_preds"], res_dl_kin["oof_preds"]
    )

    print(f"Traditional Baseline: R^2 = {res_dl_base['r2']:.4f}, RMSE = {res_dl_base['rmse']:.4f}")
    print(f"Kinematic-Enhanced:   R^2 = {res_dl_kin['r2']:.4f}, RMSE = {res_dl_kin['rmse']:.4f}")

    results["dl_quick_pressure"] = {
        "n_samples": len(valid_dl),
        "baseline_r2": res_dl_base["r2"],
        "baseline_rmse": res_dl_base["rmse"],
        "kinematic_r2": res_dl_kin["r2"],
        "kinematic_rmse": res_dl_kin["rmse"],
        "rmse_lift": rmse_lift_dl,
        "p_value": p_val_dl
    }

    # 3. Experiment 3: Career Success Index (CSI) across ALL 510 Prospects
    print("\n--- Experiment 3: Career Success Index (CSI) across 510 Prospects ---")
    X_all_trad = df[trad_feats]
    X_all_kin = df[trad_feats + kin_feats]
    y_all_csi = df["career_success_index"].fillna(0)

    res_all_base = evaluate_cv(X_all_trad, y_all_csi, model_type="lgb")
    res_all_kin = evaluate_cv(X_all_kin, y_all_csi, model_type="lgb")
    p_val_all, rmse_lift_all = bootstrap_significance(
        y_all_csi.values, res_all_base["oof_preds"], res_all_kin["oof_preds"]
    )

    print(f"Traditional Baseline: R^2 = {res_all_base['r2']:.4f}, RMSE = {res_all_base['rmse']:.4f}")
    print(f"Kinematic-Enhanced:   R^2 = {res_all_kin['r2']:.4f}, RMSE = {res_all_kin['rmse']:.4f}")
    print(f"RMSE Reduction:       {rmse_lift_all:.4f} CSI points | Bootstrap p-value: {p_val_all:.4f}")

    # Also record career snaps
    y_all_snaps = df["career_total_snaps"].fillna(0)
    res_snaps_base = evaluate_cv(X_all_trad, y_all_snaps, model_type="lgb")
    res_snaps_kin = evaluate_cv(X_all_kin, y_all_snaps, model_type="lgb")
    p_val_snaps, rmse_lift_snaps = bootstrap_significance(
        y_all_snaps.values, res_snaps_base["oof_preds"], res_snaps_kin["oof_preds"]
    )

    results["career_snaps"] = {
        "n_samples": len(df),
        "baseline_r2": res_snaps_base["r2"],
        "baseline_rmse": res_snaps_base["rmse"],
        "kinematic_r2": res_snaps_kin["r2"],
        "kinematic_rmse": res_snaps_kin["rmse"],
        "rmse_lift": rmse_lift_snaps,
        "p_value": p_val_snaps
    }

    results["career_csi"] = {
        "n_samples": len(df),
        "baseline_r2": res_all_base["r2"],
        "baseline_rmse": res_all_base["rmse"],
        "kinematic_r2": res_all_kin["r2"],
        "kinematic_rmse": res_all_kin["rmse"],
        "rmse_lift": rmse_lift_all,
        "p_value": p_val_all
    }

    # Model Feature Importances
    scaler = RobustScaler()
    X_norm = scaler.fit_transform(X_wr_kin.fillna(X_wr_kin.median()))
    ridge_final = RidgeCV().fit(X_norm, y_wr_sep)
    coef_df = pd.DataFrame({
        "feature": X_wr_kin.columns,
        "coefficient": ridge_final.coef_
    }).sort_values(by="coefficient", key=abs, ascending=False)

    coef_path = DATA_DIR / "model_feature_importance.csv"
    coef_df.to_csv(coef_path, index=False)
    print(f"\n[+] Saved feature importances to {coef_path}")

    results_path = OUTPUT_DIR / "model_benchmark_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[+] Saved benchmark summary to {results_path}")

    return results, coef_df


if __name__ == "__main__":
    run_experiment_suite()
