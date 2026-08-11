# Titanic Survival Prediction - Advanced Machine Learning Capstone


# 1. Import Libraries

import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.datasets import fetch_openml
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    RandomizedSearchCV
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

from xgboost import XGBClassifier
from catboost import CatBoostClassifier


# Project configuration.
RANDOM_STATE = 42

sns.set_theme(
    style="whitegrid",
    palette="deep"
)

# Make folders to store images and output dataset
os.makedirs("images", exist_ok=True)
os.makedirs("output", exist_ok=True)

# 2. Load and Understand Data

titanic = fetch_openml(
    name="titanic",
    version=1,
    as_frame=True
)

df = titanic.frame.copy()

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows")
print(df.head())

print("\nColumn Types")
print(df.dtypes)

print("\nDataset Information")
df.info()

print("\nStatistical Summary")
print(df.describe(include="all").T)

print("\nMissing Values")
print(df.isna().sum().sort_values(ascending=False))

print("\nDuplicate Rows:", df.duplicated().sum())


# Convert the target to integer format.
df["survived"] = pd.to_numeric(
    df["survived"]
).astype(int)


# Remove duplicate records.
df = df.drop_duplicates().copy()


# Remove leakage-prone and unsuitable raw features.
drop_columns = [
    "boat",      # Drop(Leak)
    "body",      # Drop(Leak)
    "cabin",     # many miss and randomish
    "ticket",    # cardinality
    "home.dest"  # cardinality
]

df = df.drop(
    columns=drop_columns,
    errors="ignore"
)


# 3. Exploratory Data Analysis

# Plot target balance.
plt.figure(figsize=(6, 4))

sns.countplot(
    data=df,
    x="survived",
    hue="survived",
    palette="Set2",
    legend=False
)

plt.title("Titanic-Survival-Distribution")
plt.xlabel("Survival")
plt.ylabel("Passenger Count")
plt.xticks([0, 1], ["Did Not Survive", "Survived"])
plt.tight_layout()
plt.savefig("images/Titanic-Survival-Distribution")

# Display target proportions.
print("\nTarget Distribution")
print(
    df["survived"]
    .value_counts(normalize=True)
    .sort_index()
)


# Plot numerical feature distributions.
numeric_eda = df.select_dtypes(
    include=np.number
).columns.drop("survived")

df[numeric_eda].hist(
    figsize=(12, 8),
    bins=25,
    edgecolor="black"
)

plt.tight_layout()
plt.suptitle("Numerical-Feature-Distributions")
plt.savefig("images/Numerical-Feature-Distributions")


# Plot missing-value percentages.
missing_pct = (
    df.isna().mean()
    .mul(100)
    .sort_values(ascending=False)
)

missing_pct = missing_pct[
    missing_pct > 0
]

plt.figure(figsize=(8, 5))

sns.barplot(
    x=missing_pct.values,
    y=missing_pct.index,
    color="steelblue"
)

plt.title("Missing-Values")
plt.xlabel("Missing Values (%)")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/Missing-Values")


# Examine numerical outliers with boxplots.
fig, axes = plt.subplots(
    2,
    3,
    figsize=(14, 8)
)

axes = axes.flatten()

for ax, column in zip(axes, numeric_eda):
    sns.boxplot(
        data=df,
        x=column,
        ax=ax
    )

    ax.set_title(column)

plt.tight_layout()
plt.suptitle("Numerical-Feature-Outliers")
plt.savefig("images/Numerical-Feature-Outliers")


# Examine survival by passenger sex.
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="sex",
    hue="survived"
)

plt.title("Survival-by-Sex")
plt.xlabel("Sex")
plt.ylabel("Passenger Count")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig("images/Survival-by-Sex")

 
# Examine survival by passenger class.
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="pclass",
    hue="survived"
)

plt.title("Survival-by-Passenger-Class")
plt.xlabel("Passenger Class")
plt.ylabel("Passenger Count")
plt.legend(title="Survived")
plt.tight_layout()
plt.savefig("images/Survival-by-Passenger-Class")

# Examine survival by embarkation location.
plt.figure(figsize=(7, 4))

sns.countplot(
    data=df,
    x="embarked",
    hue="survived"
)

plt.tight_layout()
plt.title("Survival-by-Embarkation-Port")
plt.savefig("images/Survival-by-Embarkation-Port")


# Examine relationships between numerical features.
correlation = df.select_dtypes(
    include=np.number
).corr()

plt.figure(figsize=(9, 6))

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.tight_layout()
plt.title("Numerical-Feature-Correlations")
plt.savefig("images/Numerical-Feature-Correlations")


# 4. Feature Engineering


# Extract passenger title from the name.
df["title"] = (
    df["name"]
    .str.extract(r",\s*([^.]*)\.")  # we do this since name has high cardinality 
    [0]                    # but we can also find useful info like mr, mrs, master, Dr, etc (Titles)
    .str.strip()
)


# Group uncommon titles into a single category.
common_titles = [
    "Mr",
    "Miss",
    "Mrs",
    "Master"
]     # if title not in common titles we mark it as rare

df["title"] = df["title"].where(
    df["title"].isin(common_titles),
    "Rare"
)


# Create family size.
df["family_size"] = (
    df["sibsp"] +
    df["parch"] +
    1
)


# Identify passengers travelling alone.
df["is_alone"] = (
    df["family_size"] == 1
).astype(int)


# Create fare per family member.
df["fare_per_person"] = (
    df["fare"] /
    df["family_size"]
)


# Remove the original name after extracting title.
df = df.drop(
    columns=["name"],
    errors="ignore"
)

print("\nFeatures After Engineering")
print(df.columns.tolist())


# 5. Split Features and Target

X = df.drop(
    columns="survived"
)

y = df["survived"]


# Split before fitting any preprocessing components.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Test Shape:", X_test.shape)


# 6. Identify Feature Types

numeric_features = X_train.select_dtypes(
    include=np.number
).columns.tolist()

categorical_features = X_train.select_dtypes(
    exclude=np.number
).columns.tolist()

print("\nNumerical Features")
print(numeric_features)

print("\nCategorical Features")
print(categorical_features)


# 7. Build Preprocessing Pipeline


# Impute and scale continuous/numerical features.
numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    )
])        


# Impute and encode categorical features.
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False
        )
    )
])


# Combine both preprocessing branches.
preprocessor = ColumnTransformer([
    (
        "numeric",
        numeric_pipeline,
        numeric_features
    ),
    (
        "categorical",
        categorical_pipeline,
        categorical_features
    )
])     # ColomnTransformer handles the traffic of the 2 categories and their pipelines


# 8. Define Candidate Models


# Logistic Regression provides an interpretable baseline.
logistic_model = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)


# Random Forest captures nonlinear relationships.
random_forest_model = RandomForestClassifier(
    n_estimators=300,
    random_state=RANDOM_STATE,
    class_weight="balanced",
    n_jobs=-1
)


# Histogram Gradient Boosting provides a sklearn boosting model.
hist_model = HistGradientBoostingClassifier(
    random_state=RANDOM_STATE
)


# XGBoost provides optimized gradient boosting.
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# CatBoost provides another strong boosting algorithm.
catboost_model = CatBoostClassifier(
    iterations=300,
    learning_rate=0.05,
    depth=5,
    verbose=0,
    random_seed=RANDOM_STATE
)


# Attach identical preprocessing to every candidate.
models = {
    "Logistic Regression": Pipeline([
        ("preprocessor", preprocessor),
        ("model", logistic_model)
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", preprocessor),
        ("model", random_forest_model)
    ]),

    "HistGradientBoosting": Pipeline([
        ("preprocessor", preprocessor),
        ("model", hist_model)
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]),

    "CatBoost": Pipeline([
        ("preprocessor", preprocessor),
        ("model", catboost_model)
    ])
}
# 5 Models made into a dict with the key as name and value as a pipeline including the model 

# 9. Cross-Validation and Model Comparison


# Use identical stratified folds for every candidate.
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE
)


# Evaluate multiple metrics during cross-validation.
scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

comparison_results = []


# Evaluate each candidate only on training data.
for model_name, model in models.items():

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

# cross_validate returns:-
# fit_time       → training time for each fold
# score_time     → prediction/scoring time for each fold
# test_accuracy  → accuracy for each fold
# test_precision → precision for each fold
# test_recall    → recall for each fold
# test_f1        → F1 score for each fold                       For reference

    comparison_results.append({
        "Model": model_name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision"].mean(),
        "Recall": scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC-AUC": scores["test_roc_auc"].mean(),
        "ROC-AUC Std": scores["test_roc_auc"].std()
    })


# Create the model leaderboard.
comparison_df = pd.DataFrame(
    comparison_results
).sort_values(
    "ROC-AUC",
    ascending=False
)

print("\nCross-Validation Model Comparison")
print(
    comparison_df.round(4)
)


# Visualize model performance.
plt.figure(figsize=(9, 5))

sns.barplot(
    data=comparison_df,
    x="ROC-AUC",
    y="Model",
    hue="Model",
    palette="viridis",
    legend=False
)

plt.title("Cross-Validated-Model-Comparison")
plt.xlabel("Mean ROC-AUC")
plt.xlim(0.5, 1)
plt.tight_layout()
plt.savefig("images/Cross-Validated-Model-Comparison")


# 10. Select Top Models

top_models = comparison_df.head(3)["Model"].tolist()

print("\nTop Three Models")
print(top_models)


# 11. Hyperparameter Tuning


# Tune Random Forest.
rf_params = {
    "model__n_estimators": [200, 300, 500, 700],
    "model__max_depth": [None, 4, 6, 8, 12],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"]
}

rf_search = RandomizedSearchCV(
    estimator=models["Random Forest"],
    param_distributions=rf_params,
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

rf_search.fit(
    X_train,
    y_train
)


# Tune XGBoost.
xgb_params = {
    "model__n_estimators": [200, 300, 500, 700],
    "model__max_depth": [2, 3, 4, 5, 6],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0]
}

xgb_search = RandomizedSearchCV(
    estimator=models["XGBoost"],
    param_distributions=xgb_params,
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

xgb_search.fit(
    X_train,
    y_train
)


# Tune CatBoost.
cat_params = {
    "model__iterations": [200, 300, 500, 700],
    "model__depth": [3, 4, 5, 6, 7],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
    "model__l2_leaf_reg": [1, 3, 5, 7, 10]
}

cat_search = RandomizedSearchCV(
    estimator=models["CatBoost"],
    param_distributions=cat_params,
    n_iter=25,
    scoring="roc_auc",
    cv=cv,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    refit=True
)

cat_search.fit(
    X_train,
    y_train
)


# Compare tuned models using training CV scores only.
tuned_results = pd.DataFrame({
    "Model": [
        "Random Forest",
        "XGBoost",
        "CatBoost"
    ],
    "Best CV ROC-AUC": [
        rf_search.best_score_,
        xgb_search.best_score_,
        cat_search.best_score_
    ]
}).sort_values(
    "Best CV ROC-AUC",
    ascending=False
)

print("\nTuned Model Results")
print(tuned_results.round(4))

print("\nRandom Forest Parameters")
print(rf_search.best_params_)

print("\nXGBoost Parameters")
print(xgb_search.best_params_)

print("\nCatBoost Parameters")
print(cat_search.best_params_)


# 12. Select Final Model


# Map tuned searches to their model names.
tuned_searches = {
    "Random Forest": rf_search,
    "XGBoost": xgb_search,
    "CatBoost": cat_search
}


# Select the strongest model based on CV performance.
best_model_name = tuned_results.iloc[0]["Model"]

final_model = tuned_searches[
    best_model_name
].best_estimator_

print("\nSelected Final Model:", best_model_name)


# 13. Test Set Evaluation


# Test data is used here for final evaluation.
y_pred = final_model.predict(
    X_test
)

y_prob = final_model.predict_proba(
    X_test
)[:, 1]


# Calculate final metrics.
test_metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "ROC-AUC": roc_auc_score(y_test, y_prob)
}

print("\nFinal Test Performance")

for metric, value in test_metrics.items():
    print(f"{metric:<10}: {value:.4f}")


print("\nClassification Report")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# 14. Confusion Matrix

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Not Survived", "Survived"],
    yticklabels=["Not Survived", "Survived"]
)

plt.title("Final-Model-Confusion-Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("images/Final-Model-Confusion-Matrix")


# 15. ROC and Precision-Recall Curves

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5)
)

RocCurveDisplay.from_predictions(
    y_test,
    y_prob,
    ax=axes[0]
)

axes[0].set_title("ROC Curve")

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_prob,
    ax=axes[1]
)

axes[1].set_title("Precision-Recall-Curve")

plt.tight_layout()
plt.savefig("images/ROC-and-PR-Curve")


# 16. Threshold Analysis


# Evaluate thresholds without retraining the model.
threshold_results = []

for threshold in np.arange(
    0.20,
    0.81,
    0.05
):

    threshold_pred = (
        y_prob >= threshold
    ).astype(int)

    threshold_results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_test,
            threshold_pred,
            zero_division=0
        ),
        "Recall": recall_score(
            y_test,
            threshold_pred,
            zero_division=0
        ),
        "F1": f1_score(
            y_test,
            threshold_pred,
            zero_division=0
        )
    })


threshold_df = pd.DataFrame(
    threshold_results
)

print("\nThreshold Analysis")
print(
    threshold_df.round(3)
)


# Plot the threshold trade-off.
plt.figure(figsize=(8, 5))

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    threshold_df["Threshold"],
    threshold_df["F1"],
    marker="o",
    label="F1"
)

plt.title("Classification Threshold Analysis")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.legend()
plt.tight_layout()
plt.savefig("images/Classification-Threshold-Analysis")


# 17. Feature Importance Using Permutation Importance


# Permutation importance works directly with the full pipeline.
permutation = permutation_importance(
    final_model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=15,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


# Importance represents the original input features.
importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": permutation.importances_mean,
    "Std": permutation.importances_std
})

importance_df = importance_df.sort_values(
    "Importance",
    ascending=False
)

print("\nPermutation Feature Importance")
print(
    importance_df.round(4)
)


# Plot permutation importance.
plt.figure(figsize=(9, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature",
    hue="Feature",
    palette="mako",
    legend=False
)

plt.title("Permutation Feature Importance")
plt.xlabel("Decrease in ROC-AUC")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("images/Permutation-data-Importance")


# 18. Error Analysis


# Build an interpretable error-analysis table.
error_analysis = X_test.copy()

error_analysis["actual"] = y_test
error_analysis["predicted"] = y_pred
error_analysis["probability"] = y_prob


# Identify incorrectly classified passengers.
errors = error_analysis[
    error_analysis["actual"] !=
    error_analysis["predicted"]
].copy()


# Measure prediction confidence.
errors["confidence"] = np.where(
    errors["predicted"] == 1,
    errors["probability"],
    1 - errors["probability"]
)


# Sort the most confident mistakes first.
errors = errors.sort_values(
    "confidence",
    ascending=False
)

print("\nNumber of Misclassified Passengers:")
print(len(errors))

print("\nMost Confident Errors:")
print(
    errors.head(15)
)


# Compare errors by passenger sex.
print("\nErrors by Sex")
print(
    errors["sex"].value_counts()
)


# Compare errors by passenger class.
print("\nErrors by Passenger Class")
print(
    errors["pclass"].value_counts()
)


# 19. Clean Prediction Output


# Create a clean deployment-style prediction file.
final_predictions = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred,
    "Survival_Probability": np.round(
        y_prob,
        4
    )
})

final_predictions.to_csv(
    "output/titanic_final_predictions.csv",
    index=False
)

print(
    "\nPredictions saved to "
    "titanic_final_predictions.csv"
)


# 20. Save Final Model


# The pipeline includes both preprocessing and prediction.
joblib.dump(
    final_model,
    "titanic_final_model.pkl"
)

print(
    "Final model saved to "
    "titanic_final_model.pkl"
)


# 21. Final Project Summary

print("\nTitanic Capstone Summary")
print("Selected Model :", best_model_name)
print(
    "CV ROC-AUC     :",
    round(
        tuned_results.iloc[0]["Best CV ROC-AUC"],
        4
    )
)

for metric, value in test_metrics.items():
    print(
        f"Test {metric:<9}: {value:.4f}"
    )

print("\nCapstone completed successfully.")