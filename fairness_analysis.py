import sys
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
    return r2, mae, rmse


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


def analyze_fairness():
    """Main function to run fairness analysis."""
    print("\n" + "="*80)
    print("FAIRNESS & BIAS ANALYSIS: Student Performance Prediction Model")
    print("="*80 + "\n")

    try:
        # Get data
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        # Scenario 1: With protected attributes (current model)
        print("Scenario 1: Model WITH Gender and Race/Ethnicity")
        print("-" * 80)
        
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        x_train = train_arr[:, :-1]
        y_train = train_arr[:, -1]
        x_test = test_arr[:, :-1]
        y_test = test_arr[:, -1]

        r2_with, mae_with, rmse_with = train_with_all_features(
            x_train, y_train, x_test, y_test
        )

        print(f"  R² Score:       {r2_with:.4f}")
        print(f"  MAE:            {mae_with:.4f}")
        print(f"  RMSE:           {rmse_with:.4f}")
        print(f"  Features:       Gender, Race/Ethnicity, Parental Education, Lunch, Test Prep, Reading & Writing Scores")

        # Scenario 2: Without protected attributes
        print("\n\nScenario 2: Model WITHOUT Gender and Race/Ethnicity")
        print("-" * 80)

        r2_without, mae_without, rmse_without = train_without_protected_attrs(
            train_df, test_df
        )

        print(f"  R² Score:       {r2_without:.4f}")
        print(f"  MAE:            {mae_without:.4f}")
        print(f"  RMSE:           {rmse_without:.4f}")
        print(f"  Features:       Parental Education, Lunch, Test Prep, Reading & Writing Scores")

        # Analysis
        print("\n\nCOMPARATIVE ANALYSIS")
        print("="*80)

        r2_delta = r2_with - r2_without
        r2_pct_change = (r2_delta / r2_without) * 100

        print(f"\nR² Score Impact:")
        print(f"  Difference:     {r2_delta:+.4f} ({r2_pct_change:+.2f}%)")
        print(f"  Performance:    {'Slightly decreased' if r2_delta < 0 else 'Slightly improved'}")

        mae_delta = mae_without - mae_with
        rmse_delta = rmse_without - rmse_with

        print(f"\nMAE Impact:")
        print(f"  Difference:     {mae_delta:+.4f}")
        print(f"  Interpretation: {'Lower is better (better without)' if mae_delta > 0 else 'Higher without (worse without)'}")

        print(f"\nRMSE Impact:")
        print(f"  Difference:     {rmse_delta:+.4f}")

        # Recommendations
        print("\n\nKEY FINDINGS & RECOMMENDATIONS")
        print("="*80)

        if abs(r2_delta) < 0.01:
            print(f"""
✓ MINIMAL ACCURACY LOSS: Removing protected attributes costs only {abs(r2_pct_change):.2f}% in R² score.
  This suggests that gender and race/ethnicity are weak predictive signals for 
  math performance, after accounting for other factors.

✓ FAIRNESS CONSIDERATION: The model can maintain nearly identical predictive power
  while eliminating protected attributes, reducing potential for discriminatory 
  predictions based on immutable characteristics.

✓ RECOMMENDATION: Use the debiased model (without protected attributes) because:
  1. Legal/Ethical: Eliminates direct use of protected characteristics
  2. Fairness: Reduces risk of perpetuating historical biases in educational outcomes
  3. Practical: Negligible impact on prediction accuracy
  4. Explainability: Easier to explain to stakeholders (focuses on behaviors/scores)

⚠ CAVEAT: Removing features doesn't eliminate all bias. Historical disparities in
  reading/writing scores may still reflect systemic inequities. True fairness 
  requires ongoing monitoring for proxy discrimination and dataset audit.
            """)
        else:
            print(f"""
✗ SIGNIFICANT ACCURACY LOSS: Removing protected attributes costs {abs(r2_pct_change):.2f}% in R² score.
  This suggests gender and race/ethnicity are meaningful predictive features.

This presents an ethical trade-off:
  • Keep features: Better accuracy but risk of discriminatory predictions
  • Remove features: Fair treatment but reduced predictive power

RECOMMENDATION: Still recommend the debiased model because:
  1. The ethical cost of using protected attributes outweighs the accuracy gain
  2. Better to be slightly less accurate than systematically biased
  3. Fairness constraints are often required by law/regulation
  4. Can improve other features (socioeconomic indicators) to recover accuracy
            """)

        print("\n" + "="*80 + "\n")

    except Exception as e:
        print(f"Error during fairness analysis: {str(e)}")
        raise CustomException(e, sys)


if __name__ == "__main__":
    analyze_fairness()
