import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd



from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
)

try:
    from xgboost import XGBClassifier
    IS_BOOST_READY = True
except Exception:
    IS_BOOST_READY = False

PRIMARY_DATASET_LOC = "DataSet.csv"
RESPONSE_VARIABLE = "TARGET_COL"

HIT_LABEL = "1"
MISS_LABEL = "0"

THRESHOLD_FEATS = 12
THRESHOLD_RECORDS = 500

SPLIT_RATIO = 0.2
UNIFORM_SEED = 42

NB_VARIANT = "gaussian"

def check_data_integrity(data: pd.DataFrame) -> None:
    rows, cols = data.shape
    if rows < THRESHOLD_RECORDS:
        raise ValueError(f"Insufficient samples: {rows} < {THRESHOLD_RECORDS}")
    if cols < THRESHOLD_FEATS:
        raise ValueError(f"Insufficient features: {cols} < {THRESHOLD_FEATS}")

def encode_binary_labels(raw_labels: pd.Series) -> pd.Series:
    refined = raw_labels.astype(str).str.strip()
    transformed = refined.replace({HIT_LABEL: 1, MISS_LABEL: 0})
    
    if transformed.isin([0, 1]).mean() < 1.0:
        lowercase_tags = refined.str.lower()
        transformed = lowercase_tags.replace({str(HIT_LABEL).lower(): 1, str(MISS_LABEL).lower(): 0})
    
    if transformed.isin([0, 1]).mean() < 1.0:
        invalid_entries = refined[~transformed.isin([0, 1])].unique()[:10]
        raise ValueError(f"Unexpected values in target: {invalid_entries}")
    
    return transformed.astype(int)

def construct_data_pipeline(data: pd.DataFrame, use_min_max: bool = False) -> ColumnTransformer:
    numerical_attrs = data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_attrs = [col for col in data.columns if col not in numerical_attrs]

    fill_numerical = SimpleImputer(strategy="median")
    fill_categorical = SimpleImputer(strategy="most_frequent")

    if use_min_max:
        from sklearn.preprocessing import MinMaxScaler
        numeric_processor = MinMaxScaler()
    else:
        numeric_processor = StandardScaler()

    numerical_flow = Pipeline(steps=[
        ("impute", fill_numerical),
        ("scale", numeric_processor),
    ])

    categorical_flow = Pipeline(steps=[
        ("impute", fill_categorical),
        ("encode", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num_proc", numerical_flow, numerical_attrs),
            ("cat_proc", categorical_flow, categorical_attrs),
        ],
        remainder="drop"
    )

def compute_model_metrics(ground_truth, predicted, probability_array) -> dict:
    auc_val = roc_auc_score(ground_truth, probability_array) if len(np.unique(ground_truth)) == 2 else np.nan
    return {
        "Accuracy": accuracy_score(ground_truth, predicted),
        "AUC": auc_val,
        "Precision": precision_score(ground_truth, predicted, zero_division=0),
        "Recall": recall_score(ground_truth, predicted, zero_division=0),
        "F1": f1_score(ground_truth, predicted, zero_division=0),
        "MCC": matthews_corrcoef(ground_truth, predicted),
    }

def fetch_prediction_probabilities(pipeline_obj, test_features):
    if hasattr(pipeline_obj, "predict_proba"):
        return pipeline_obj.predict_proba(test_features)[:, 1]
    if hasattr(pipeline_obj, "decision_function"):
        raw_vals = pipeline_obj.decision_function(test_features)
        v_min, v_max = np.min(raw_vals), np.max(raw_vals)
        if v_max == v_min:
            return np.zeros_like(raw_vals, dtype=float)
        return (raw_vals - v_min) / (v_max - v_min)
    return pipeline_obj.predict(test_features).astype(float)

def display_metrics_summary(evaluation_data: dict):
    performance_keys = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    print("\n" + "="*20 + " PERFORMANCE REPORT " + "="*20)
    top_row = f"{'Algorithm':30s}" + "".join([f"{key:>12s}" for key in performance_keys])
    print(top_row)
    print("-" * len(top_row))
    for algo_name, stats in evaluation_data.items():
        data_row = f"{algo_name:30s}" + "".join([f"{stats[key]:12.6f}" for key in performance_keys])
        print(data_row)
    print("="*60 + "\n")

import joblib
import os
import cloudpickle

def convert_to_dense(matrix_data):
    return matrix_data.toarray() if hasattr(matrix_data, "toarray") else matrix_data

def execute_training_cycle():
    if not os.path.exists("model"):
        os.makedirs("model")

    dataset = pd.read_csv(PRIMARY_DATASET_LOC)

    if RESPONSE_VARIABLE not in dataset.columns:
        raise KeyError(f"Missing outcome field: '{RESPONSE_VARIABLE}'")

    features = dataset.drop(columns=[RESPONSE_VARIABLE])
    labels = encode_binary_labels(dataset[RESPONSE_VARIABLE])

    check_data_integrity(features)

    train_x, test_x, train_y, test_y = train_test_split(
        features, labels,
        test_size=SPLIT_RATIO,
        random_state=UNIFORM_SEED,
        stratify=labels
    )

    summary_stats = {}

    default_prep = construct_data_pipeline(train_x, use_min_max=False)

    logistic_clf = Pipeline(steps=[
        ("prep", default_prep),
        ("clf", LogisticRegression(max_iter=5000, solver="lbfgs"))
    ])

    tree_clf = Pipeline(steps=[
        ("prep", default_prep),
        ("clf", DecisionTreeClassifier(random_state=UNIFORM_SEED))
    ])

    neighbor_clf = Pipeline(steps=[
        ("prep", default_prep),
        ("clf", KNeighborsClassifier(n_neighbors=5))
    ])

    if NB_VARIANT.lower() == "multinomial":
        nb_prep = construct_data_pipeline(train_x, use_min_max=True)
        bayes_clf = Pipeline(steps=[
            ("prep", nb_prep),
            ("clf", MultinomialNB())
        ])
        bayes_label = "Naive Bayes (MN)"
    else:
        from sklearn.preprocessing import FunctionTransformer
        bayes_clf = Pipeline(steps=[
            ("prep", default_prep),
            ("dense", FunctionTransformer(convert_to_dense, accept_sparse=True)),
            ("clf", GaussianNB())
        ])
        bayes_label = "Naive Bayes"

    forest_clf = Pipeline(steps=[
        ("prep", default_prep),
        ("clf", RandomForestClassifier(
            n_estimators=300,
            random_state=UNIFORM_SEED,
            n_jobs=-1
        ))
    ])

    if IS_BOOST_READY:
        boost_clf = Pipeline(steps=[
            ("prep", default_prep),
            ("clf", XGBClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=1.0,
                random_state=UNIFORM_SEED,
                eval_metric="logloss",
                n_jobs=-1
            ))
        ])
        boost_label = "XGBoost"
    else:
        boost_clf = None
        boost_label = "XGBoost (unavailable)"

    candidate_models = [
        ("Logistic Regression", logistic_clf),
        ("Decision Tree", tree_clf),
        ("KNN", neighbor_clf),
        (bayes_label, bayes_clf),
        ("Random Forest", forest_clf),
    ]
    if boost_clf is not None:
        candidate_models.append((boost_label, boost_clf))

    for tag, pipe in candidate_models:
        print(f"Fitting {tag}...")
        pipe.fit(train_x, train_y)
        preds = pipe.predict(test_x)
        probs = fetch_prediction_probabilities(pipe, test_x)
        summary_stats[tag] = compute_model_metrics(test_y, preds, probs)
        
        target_path = f"model/{tag.lower().replace(' ', '_').replace('(', '').replace(')', '')}.joblib"
        joblib.dump(pipe, target_path)
        print(f"Archived {tag} at {target_path}")
        # Also save a cloudpickle copy for improved portability between environments
        try:
            cp_path = target_path.replace('.joblib', '.cpkl')
            with open(cp_path, 'wb') as _f:
                cloudpickle.dump(pipe, _f)
            print(f"Also archived (cloudpickle) {tag} at {cp_path}")
        except Exception:
            print("Warning: failed to write cloudpickle copy for", tag)

    display_metrics_summary(summary_stats)

if __name__ == "__main__":
    execute_training_cycle()