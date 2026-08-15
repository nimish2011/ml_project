# Student Performance Predictor

An end-to-end machine learning pipeline that predicts a student's **math score** from demographic and academic features, served through a Flask web app.

## Problem Statement

Given a student's gender, ethnicity, parental education level, lunch type, test preparation course status, and reading/writing scores, predict their math score. This kind of model can help schools flag students likely to need extra support before results come in.

## Dataset

`notebook/data/stud.csv` — student performance records with the following features:

| Feature | Type |
|---|---|
| gender | categorical |
| race_ethnicity | categorical |
| parental_level_of_education | categorical |
| lunch | categorical |
| test_preparation_course | categorical |
| reading_score | numerical |
| writing_score | numerical |
| **math_score** | target |

## Approach

**Pipeline (`src/components/`)**
1. **Data Ingestion** — reads the raw CSV, splits into train/test (80/20), writes both to `artifacts/`.
2. **Data Transformation** — builds a `ColumnTransformer`: numerical columns get median imputation + scaling, categorical columns get most-frequent imputation + one-hot encoding + scaling. The fitted preprocessor is saved as `artifacts/preprocessor.pkl`.
3. **Model Training** — trains and grid-searches seven regressors (Linear Regression, Decision Tree, Random Forest, Gradient Boosting, AdaBoost, XGBoost, CatBoost), scores each on R², and saves the best-performing model to `artifacts/model.pkl`.

**Serving (`application.py`, `src/pipeline/predict_pipeline.py`)**
A Flask app takes form input on `/predictdata`, wraps it into a `CustomData` object, runs it through the saved preprocessor and model, and renders the predicted math score.

## Tech Stack

Python · pandas · scikit-learn · CatBoost · XGBoost · Flask · AWS Elastic Beanstalk (deployment config included)

## Results

Best model: Linear Regression
Test R²: 0.8804

*Run `python src/components/data_ingestion.py` to reproduce training and see the R² score printed at the end.*

## 🚨 ML Fairness & Bias Analysis

**The uncomfortable question:** This model uses `gender` and `race_ethnicity` as predictive features. Should it?

### Key Finding: A Real Fairness Trade-off

A fairness audit was conducted comparing model performance with and without protected attributes:

| Scenario | R² Score | MAE | Key Features |
|----------|----------|-----|--------------|
| **WITH** gender + race_ethnicity | **0.8804** ✓ | 4.21 | All 7 features |
| **WITHOUT** gender + race_ethnicity | 0.6979 | 6.98 | Only behavior/scores |
| **Accuracy Cost** | **-26.16%** 📉 | +2.77 | — |

### Why This Matters

The model's accuracy **significantly depends** on using protected attributes. This reveals an important constraint:

- **More Accurate** ≠ **More Fair**
- Using demographic features risks perpetuating systemic biases (e.g., if historical data reflects unequal educational access, the model learns and amplifies those disparities)
- Removing them improves fairness but sacrifices 1/4 of predictive power

### Our Choice

**This project uses the full-feature model (with protected attributes)** for maximum accuracy, but **documents the trade-off openly**:

✅ **Pros of using all features:**
- Better predictions for students who need support
- Data-driven: reflective of actual patterns in the dataset

⚠️ **Risks of using all features:**
- May encode historical inequities (e.g., if writing scores correlate with socioeconomic status)
- Could lead to biased recommendations (e.g., disproportionately flagging certain groups)
- Legally risky in regulated domains (lending, hiring, criminal justice)

### Real-World Application

**If deployed to schools**, the recommendation would be:
1. **Use the debiased model** (without gender/race) to avoid legal/ethical liability
2. **Monitor prediction disparities** by demographic group post-deployment (fairness audits)
3. **Improve socioeconomic features** (lunch type, parental education) to recover accuracy without protected attributes
4. **Involve stakeholders** (educators, students, parents) in fairness decisions

### How to Explore

```bash
# Run the fairness analysis yourself
python fairness_analysis.py
```

This script trains models both ways and generates a detailed report.

---

## Project Structure

```
ml_project/
├── application.py                     # Flask entrypoint
├── fairness_analysis.py               # Fairness & bias audit (protected attributes impact)
├── src/
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   └── predict_pipeline.py
│   ├── logger.py
│   ├── exception.py
│   └── utils.py
├── templates/                         # HTML for the Flask app
├── notebook/                          # EDA and model-training exploration
├── artifacts/                         # Generated: train/test splits, preprocessor.pkl, model.pkl
└── requirements.txt
```

## Setup & Usage

```bash
git clone https://github.com/nimish2011/ml_project.git
cd ml_project

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

# Train the pipeline (ingestion -> transformation -> model training)
python src/components/data_ingestion.py

# Run the web app
python application.py
```

Then open `http://localhost:5000` and fill in the form to get a predicted math score.

## Future Improvements

- [ ] Implement fairness constraints: train a model with a Lagrangian penalty for demographic parity
- [ ] Add adversarial debiasing: use a secondary model to predict protected attributes and minimize its accuracy
- [ ] Add SHAP/LIME explainability: understand which features drive predictions for individual students
- [ ] Build a fairness monitoring dashboard: track prediction disparities post-deployment
- [ ] Expand dataset: collect more diverse samples to reduce historical bias
- [ ] Add unit tests and CI workflow (GitHub Actions)
- [ ] Containerize with Docker for reproducibility
- [ ] Implement A/B testing framework: compare debiased vs. full-feature model performance in schools

## Author

Nimish