"""
Automated Pipeline and Data Quality Unit Tests for NFL Big Data Bowl 2027.
Verifies data integrity, kinematic feature extraction, target aggregation,
benchmarking results, visualization assets, and submission notebook formatting.
"""

import json
import unittest
from pathlib import Path
import pandas as pd
import nbformat as nbf

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "eda_outputs"
NOTEBOOK_DIR = BASE_DIR / "notebooks"


class TestNFLBigDataBowlPipeline(unittest.TestCase):

    def test_01_raw_data_integrity(self):
        """Verify presence and row counts of essential raw datasets."""
        files = {
            "players.csv": 500,
            "combine_results.csv": 500,
            "player_career_successes.csv": 500,
            "games.csv": 1000
        }
        for fname, min_rows in files.items():
            fpath = DATA_DIR / fname
            self.assertTrue(fpath.exists(), f"Missing file: {fname}")
            df = pd.read_csv(fpath)
            self.assertGreaterEqual(len(df), min_rows, f"File {fname} has fewer than {min_rows} rows.")

    def test_02_kinematic_features_integrity(self):
        """Verify extracted prospect_kinematics.csv features."""
        kin_path = DATA_DIR / "prospect_kinematics.csv"
        self.assertTrue(kin_path.exists(), "prospect_kinematics.csv does not exist.")
        df = pd.read_csv(kin_path)
        self.assertEqual(len(df), 510, "Expected 510 prospects in kinematics matrix.")

        expected_cols = [
            "nfl_id", "comb_peak_speed", "comb_peak_accel", "comb_mean_abs_jerk",
            "route_kbe", "route_rsi", "route_cut_burst_accel",
            "trench_bend_index", "trench_first_step_burst"
        ]
        for col in expected_cols:
            self.assertIn(col, df.columns, f"Missing kinematic column: {col}")
            self.assertFalse(df[col].isnull().all(), f"Kinematic column {col} is completely null.")

    def test_03_nfl_targets_integrity(self):
        """Verify compiled nfl_performance_targets.csv targets."""
        target_path = DATA_DIR / "nfl_performance_targets.csv"
        self.assertTrue(target_path.exists(), "nfl_performance_targets.csv does not exist.")
        df = pd.read_csv(target_path)
        self.assertEqual(len(df), 510, "Expected 510 prospects in targets dataset.")

        expected_targets = [
            "nfl_id", "wr_mean_separation", "career_total_snaps",
            "career_games_started", "dl_quick_pressure_rate"
        ]
        for col in expected_targets:
            self.assertIn(col, df.columns, f"Missing target column: {col}")

    def test_04_model_benchmark_results(self):
        """Verify machine learning benchmark results and kinematic superiority."""
        res_path = OUTPUT_DIR / "model_benchmark_results.json"
        self.assertTrue(res_path.exists(), "model_benchmark_results.json does not exist.")
        with open(res_path, "r") as f:
            bench = json.load(f)

        self.assertIn("wr_separation", bench)
        self.assertIn("career_snaps", bench)

        # Kinematic R^2 must exceed baseline R^2 on WR separation
        self.assertGreater(
            bench["wr_separation"]["kinematic_r2"],
            bench["wr_separation"]["baseline_r2"],
            "Kinematic model failed to outperform baseline on WR separation."
        )

        # Kinematic R^2 must exceed baseline R^2 on career total snaps
        self.assertGreater(
            bench["career_snaps"]["kinematic_r2"],
            bench["career_snaps"]["baseline_r2"],
            "Kinematic model failed to outperform baseline on career total snaps."
        )
        self.assertLessEqual(
            bench["career_snaps"]["p_value"],
            0.05,
            "Career snaps improvement was not statistically significant at p < 0.05."
        )

    def test_05_visualizations_exist(self):
        """Verify all 7 figures exist and are non-empty."""
        figs = [
            "figure1_combine_missingness.png",
            "figure2_traditional_vs_nfl_correlation.png",
            "figure3_wr_separation_regression.png",
            "figure4_model_performance_comparison.png",
            "figure5_feature_importance_coefficients.png",
            "figure6_drill_kinematics_trajectory.png",
            "figure7_prospect_scout_radar.png"
        ]
        for fig_name in figs:
            fig_path = OUTPUT_DIR / fig_name
            self.assertTrue(fig_path.exists(), f"Figure missing: {fig_name}")
            self.assertGreater(fig_path.stat().st_size, 1000, f"Figure {fig_name} is too small / empty.")

    def test_06_submission_notebook(self):
        """Verify structure and constraints of master Jupyter notebook."""
        nb_path = NOTEBOOK_DIR / "nfl_big_data_bowl_2027_solution.ipynb"
        self.assertTrue(nb_path.exists(), "Master notebook does not exist.")
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        self.assertGreater(len(nb.cells), 5, "Notebook has too few cells.")
        # Ensure code cells and markdown cells are properly formatted
        code_cells = [c for c in nb.cells if c.cell_type == "code"]
        md_cells = [c for c in nb.cells if c.cell_type == "markdown"]
        self.assertGreater(len(code_cells), 2)
        self.assertGreater(len(md_cells), 3)


if __name__ == "__main__":
    unittest.main()
