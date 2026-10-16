"""
Exploratory Data Analysis (EDA) for NFL Big Data Bowl 2027.
Investigates prospect demographics, Combine drill completion rates,
missingness profiles, and baseline correlations with professional success.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_datasets():
    """Load core metadata and summary datasets."""
    players = pd.read_csv(DATA_DIR / "players.csv")
    combine_results = pd.read_csv(DATA_DIR / "combine_results.csv")
    career = pd.read_csv(DATA_DIR / "player_career_successes.csv")
    games = pd.read_csv(DATA_DIR / "games.csv")
    return players, combine_results, career, games


def analyze_missingness(combine_results: pd.DataFrame) -> pd.DataFrame:
    """Analyze drill skipping / missing rates in traditional Combine testing."""
    drill_cols = [
        "forty", "ten_yd_split", "vertical", "broad_jump",
        "three_cone", "short_shuttle", "bench_reps"
    ]
    missing_stats = []
    total_prospects = len(combine_results)

    for col in drill_cols:
        missing_count = combine_results[col].isnull().sum()
        pct = (missing_count / total_prospects) * 100
        missing_stats.append({
            "drill": col,
            "completed": total_prospects - missing_count,
            "missing": missing_count,
            "missing_pct": pct
        })

    df_missing = pd.DataFrame(missing_stats).sort_values("missing_pct", ascending=False)
    return df_missing


def plot_missingness(df_missing: pd.DataFrame, save_path: Path = None):
    """Generate Figure 1: Traditional Combine Missingness & Drill Skipping."""
    plt.figure(figsize=(9, 5), dpi=150)
    sns.set_theme(style="whitegrid")
    
    palette = sns.color_palette("Reds_r", len(df_missing))
    bars = plt.barh(df_missing["drill"], df_missing["missing_pct"], color=palette, edgecolor="black")

    for bar in bars:
        width = bar.get_width()
        plt.text(
            width + 1.2, bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center", ha="left", fontsize=10, fontweight="bold", color="#333333"
        )

    plt.title("Traditional NFL Combine: High Drill Skipping & Missingness Rate", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Percentage of Prospects Missing Test Data (%)", fontsize=11)
    plt.ylabel("Combine Drill", fontsize=11)
    plt.xlim(0, 100)
    plt.axvline(50, color="gray", linestyle="--", alpha=0.6, label="50% Missing Threshold")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[+] Saved missingness plot to {save_path}")
    plt.close()


def analyze_traditional_correlations(players: pd.DataFrame, combine: pd.DataFrame, career: pd.DataFrame):
    """Examine correlation between traditional metrics and regular season success."""
    merged = players.merge(combine, on="nfl_id").merge(career, on="nfl_id")

    # Filter to skill pass-catchers (WR)
    wr_df = merged[merged["nfl_position"] == "WR"].copy()
    
    metrics = [
        "forty", "ten_yd_split", "vertical", "broad_jump",
        "ngs_athleticism_score", "ngs_college_production_score"
    ]
    targets = ["career_offensive_snaps", "career_games_started"]

    corr_matrix = wr_df[metrics + targets].corr().loc[metrics, targets]
    return wr_df, corr_matrix


def plot_correlation_breakdown(corr_matrix: pd.DataFrame, save_path: Path = None):
    """Plot correlations of traditional Combine metrics with NFL career outcomes."""
    plt.figure(figsize=(8, 4.5), dpi=150)
    sns.set_theme(style="white")
    
    ax = sns.heatmap(
        corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", center=0,
        cbar_kws={"label": "Pearson Correlation (r)"}, linewidths=1, linecolor="white"
    )
    plt.title("Correlation: Traditional Combine Metrics vs NFL Career Snaps / Starts (WRs)", fontsize=12, fontweight="bold", pad=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[+] Saved correlation plot to {save_path}")
    plt.close()


def run_eda():
    """Execute complete EDA and output summary statistics."""
    print("=== NFL Big Data Bowl 2027: Exploratory Data Analysis ===")
    players, combine, career, games = load_datasets()

    print(f"[+] Total prospects: {len(players)}")
    print(f"[+] Draft classes: {sorted(players['draft_year'].unique().tolist())}")
    print(f"[+] Total regular season games tracked: {len(games)}")

    df_missing = analyze_missingness(combine)
    print("\nCombine Missingness Summary:")
    print(df_missing.to_string(index=False))

    plot_missingness(df_missing, OUTPUT_DIR / "figure1_combine_missingness.png")

    _, wr_corr = analyze_traditional_correlations(players, combine, career)
    print("\nWR Traditional Metrics Correlation with NFL Career Outcomes:")
    print(wr_corr)

    plot_correlation_breakdown(wr_corr, OUTPUT_DIR / "figure2_traditional_vs_nfl_correlation.png")
    print("\n[+] EDA completed successfully.")


if __name__ == "__main__":
    run_eda()
