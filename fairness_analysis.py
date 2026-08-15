import sys
import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException


def train_with_all_features(x_train, y_train, x_test, y_test):
    """Train model with all features including protected attributes."""
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return model, y_pred, r2, mae, rmse


def get_data_transformer_without_protected_attrs():
    """Create preprocessing pipeline excluding gender and race_ethnicity."""
    numerical_columns = ["writing_score", "reading_score"]
    categorical_columns = [
        "parental_level_of_education",
        "lunch",
        "test_preparation_course"
        # Removed: "gender", "race_ethnicity"
    ]

    num_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False))
        ]
    )

    cat_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
            ("scaler", StandardScaler(with_mean=False))
        ]
    )

    preprocessor = ColumnTransformer([
        ("num_pipeline", num_pipeline, numerical_columns),
        ("cat_pipeline", cat_pipeline, categorical_columns)
    ])

    return preprocessor


def train_without_protected_attrs(train_df, test_df):
    """Train model after removing protected attributes."""
    target_column_name = "math_score"

    # Drop protected attributes
    input_feature_train_df = train_df.drop(
        columns=[target_column_name, "gender", "race_ethnicity"]
    )
    target_feature_train_df = train_df[target_column_name]

    input_feature_test_df = test_df.drop(
        columns=[target_column_name, "gender", "race_ethnicity"]
    )
    target_feature_test_df = test_df[target_column_name]

    # Get transformer without protected attributes
    preprocessor = get_data_transformer_without_protected_attrs()

    # Transform data
    x_train = preprocessor.fit_transform(input_feature_train_df)
    x_test = preprocessor.transform(input_feature_test_df)

    y_train = target_feature_train_df.values
    y_test = target_feature_test_df.values

    # Train model
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return r2, mae, rmse


def group_wise_error(test_df, y_test, y_pred, group_col):
    """
    Compute mean absolute error per subgroup to check for disparate
    error rates -- i.e. does the model perform worse for some groups
    than others, even if overall accuracy looks fine?
    """
    results_df = test_df.reset_index(drop=True).copy()
    results_df["y_test"] = np.asarray(y_test)
    results_df["y_pred"] = np.asarray(y_pred)
    results_df["abs_error"] = np.abs(results_df["y_test"] - results_df["y_pred"])

    summary = results_df.groupby(group_col)["abs_error"].agg(["mean", "count"])
    summary = summary.rename(columns={"mean": "mae", "count": "n"})
    summary = summary.sort_values("mae", ascending=False)
    return summary


def print_group_wise_error(test_df, y_test, y_pred, group_col):
    """Pretty-print the group-wise MAE table and flag the largest gap."""
    summary = group_wise_error(test_df, y_test, y_pred, group_col)

    print(f"\nGroup-wise MAE by '{group_col}':")
    print("-" * 80)
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    max_mae = summary["mae"].max()
    min_mae = summary["mae"].min()
    gap = max_mae - min_mae
    worst_group = summary["mae"].idxmax()
    best_group = summary["mae"].idxmin()

    print(f"\n  Largest MAE gap:  {gap:.4f} points")
    print(f"  Highest error:    '{worst_group}' (MAE = {max_mae:.4f})")
    print(f"  Lowest error:     '{best_group}' (MAE = {min_mae:.4f})")

    return summary, gap


def analyze_fairness(save_report=True):
    """Main function to run fairness analysis."""
    report_lines = []

    def log(msg=""):
        print(msg)
        report_lines.append(str(msg))

    log("\n" + "=" * 80)
    log("FAIRNESS & BIAS ANALYSIS: Student Performance Prediction Model")
    log("=" * 80 + "\n")

    try:
        # Get data
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # ------------------------------------------------------------------
        # Scenario 1: With protected attributes (current model)
        # ------------------------------------------------------------------
        log("Scenario 1: Model WITH Gender and Race/Ethnicity")
        log("-" * 80)

        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        x_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]
        x_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        model_with, y_pred_with, r2_with, mae_with, rmse_with = train_with_all_features(
            x_train, y_train, x_test, y_test
        )

        log(f"  R2 Score:       {r2_with:.4f}")
        log(f"  MAE:            {mae_with:.4f}")
        log(f"  RMSE:           {rmse_with:.4f}")
        log(f"  Features:       Gender, Race/Ethnicity, Parental Education, Lunch, Test Prep, Reading & Writing Scores")

        # ------------------------------------------------------------------
        # Group-wise error breakdown (the actual fairness check)
        # ------------------------------------------------------------------
        log("\n\nDISPARATE ERROR ANALYSIS (on the model WITH protected attributes)")
        log("=" * 80)
        log("Overall accuracy can look fine while errors are unevenly distributed")
        log("across groups. This checks per-subgroup MAE to see if that's happening.")

        gender_summary, gender_gap = print_group_wise_error(
            test_df, y_test, y_pred_with, "gender"
        )
        report_lines.append(gender_summary.to_string(float_format=lambda x: f"{x:.4f}"))

        race_summary, race_gap = print_group_wise_error(
            test_df, y_test, y_pred_with, "race_ethnicity"
        )
        report_lines.append(race_summary.to_string(float_format=lambda x: f"{x:.4f}"))

        # ------------------------------------------------------------------
        # Scenario 2: Without protected attributes
        # ------------------------------------------------------------------
        log("\n\nScenario 2: Model WITHOUT Gender and Race/Ethnicity")
        log("-" * 80)

        r2_without, mae_without, rmse_without = train_without_protected_attrs(
            train_df, test_df
        )

        log(f"  R2 Score:       {r2_without:.4f}")
        log(f"  MAE:            {mae_without:.4f}")
        log(f"  RMSE:           {rmse_without:.4f}")
        log(f"  Features:       Parental Education, Lunch, Test Prep, Reading & Writing Scores")

        # ------------------------------------------------------------------
        # Comparative analysis
        # ------------------------------------------------------------------
        log("\n\nCOMPARATIVE ANALYSIS")
        log("=" * 80)

        r2_delta = r2_with - r2_without
        r2_pct_change = (r2_delta / r2_without) * 100

        log(f"\nR2 Score Impact:")
        log(f"  Difference:     {r2_delta:+.4f} ({r2_pct_change:+.2f}%)")
        log(f"  Performance:    {'Slightly decreased' if r2_delta < 0 else 'Slightly improved'}")

        mae_delta = mae_without - mae_with
        rmse_delta = rmse_without - rmse_with

        log(f"\nMAE Impact:")
        log(f"  Difference:     {mae_delta:+.4f}")
        log(f"  Interpretation: {'Lower is better (better without)' if mae_delta > 0 else 'Higher without (worse without)'}")

        log(f"\nRMSE Impact:")
        log(f"  Difference:     {rmse_delta:+.4f}")

        # ------------------------------------------------------------------
        # Key findings
        # ------------------------------------------------------------------
        log("\n\nKEY FINDINGS & RECOMMENDATIONS")
        log("=" * 80)

        if abs(r2_delta) < 0.01:
            log(f"""
[OK] MINIMAL ACCURACY LOSS: Removing protected attributes costs only {abs(r2_pct_change):.2f}% in R2 score.
  This suggests that gender and race/ethnicity are weak predictive signals for
  math performance, after accounting for other factors.

[OK] FAIRNESS CONSIDERATION: The model can maintain nearly identical predictive power
  while eliminating protected attributes, reducing potential for discriminatory
  predictions based on immutable characteristics.

[OK] RECOMMENDATION: Use the debiased model (without protected attributes) because:
  1. Legal/Ethical: Eliminates direct use of protected characteristics
  2. Fairness: Reduces risk of perpetuating historical biases in educational outcomes
  3. Practical: Negligible impact on prediction accuracy
  4. Explainability: Easier to explain to stakeholders (focuses on behaviors/scores)

[!] CAVEAT: Removing features doesn't eliminate all bias. Historical disparities in
  reading/writing scores may still reflect systemic inequities. True fairness
  requires ongoing monitoring for proxy discrimination and dataset audit.
            """)
        else:
            log(f"""
[X] SIGNIFICANT ACCURACY LOSS: Removing protected attributes costs {abs(r2_pct_change):.2f}% in R2 score.
  This suggests gender and race/ethnicity are meaningful predictive features.

This presents an ethical trade-off:
  - Keep features: Better accuracy but risk of discriminatory predictions
  - Remove features: Fair treatment but reduced predictive power

RECOMMENDATION: Still recommend the debiased model because:
  1. The ethical cost of using protected attributes outweighs the accuracy gain
  2. Better to be slightly less accurate than systematically biased
  3. Fairness constraints are often required by law/regulation
  4. Can improve other features (socioeconomic indicators) to recover accuracy
            """)

        # ------------------------------------------------------------------
        # Disparate error findings
        # ------------------------------------------------------------------
        log("\nDISPARATE ERROR FINDINGS")
        log("-" * 80)
        log(f"  Gender MAE gap:          {gender_gap:.4f} points")
        log(f"  Race/Ethnicity MAE gap:  {race_gap:.4f} points")

        max_gap = max(gender_gap, race_gap)
        if max_gap >= 2.0:
            log(f"""
[!] NOTABLE DISPARITY: The largest per-group MAE gap is {max_gap:.4f} points, which is
  large enough to matter in practice (e.g. affecting which students get flagged
  for support). Even though removing protected attributes barely changes overall
  R2, the model's errors are NOT evenly distributed across groups. This is the
  kind of gap aggregate accuracy metrics hide -- worth investigating further
  (e.g. is one subgroup underrepresented in training data?).
            """)
        else:
            log(f"""
[OK] No large disparity detected: the largest per-group MAE gap ({max_gap:.4f} points)
  is small relative to overall error. No strong evidence of the model performing
  meaningfully worse for any single subgroup in this dataset.
            """)

        log("\n" + "=" * 80 + "\n")

        if save_report:
            os.makedirs("reports", exist_ok=True)
            report_path = os.path.join("reports", "fairness_analysis.txt")
            with open(report_path, "w") as f:
                f.write("\n".join(report_lines))
            print(f"Report saved to {report_path}")

    except Exception as e:
        print(f"Error during fairness analysis: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    analyze_fairness()