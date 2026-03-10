import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def load_data():
    X = pd.read_csv('data/raw/X_processed.csv')
    y = pd.read_csv('data/raw/y.csv').squeeze()
    return X, (y == 1).astype(int)

def run_experiment(X, y, use_smote=False, use_pca=False, use_threshold=False, n_components=50):
    X_np = StandardScaler().fit_transform(X)

    if use_pca:
        X_np = PCA(n_components=n_components, random_state=42).fit_transform(X_np)

    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y, test_size=0.2, stratify=y, random_state=42
    )

    if use_smote:
        X_train, y_train = SMOTE(random_state=42, k_neighbors=3).fit_resample(X_train, y_train)

    if use_threshold:
        sw = np.where(y_train == 1, 8, 1)
        model = GradientBoostingClassifier(n_estimators=300, max_depth=3,
                    learning_rate=0.05, subsample=0.8, random_state=42)
        model.fit(X_train, y_train, sample_weight=sw)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.2).astype(int)
    else:
        model = GradientBoostingClassifier(n_estimators=200, max_depth=4,
                    learning_rate=0.05, random_state=42)
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

    return {
        'Precision': round(precision_score(y_test, y_pred, zero_division=0), 3),
        'Recall':    round(recall_score(y_test, y_pred, zero_division=0), 3),
        'F1':        round(f1_score(y_test, y_pred, zero_division=0), 3),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 3),
    }

if __name__ == "__main__":
    X, y = load_data()

    experiments = [
        ("Baseline (GBM)",                        dict(use_smote=False, use_pca=False, use_threshold=False)),
        ("+ SMOTE",                               dict(use_smote=True,  use_pca=False, use_threshold=False)),
        ("+ SMOTE + PCA",                         dict(use_smote=True,  use_pca=True,  use_threshold=False)),
        ("+ SMOTE + PCA + Threshold Tuning",      dict(use_smote=True,  use_pca=True,  use_threshold=True)),
    ]

    results = []
    for name, kwargs in experiments:
        print(f"Running: {name}...")
        r = run_experiment(X, y, **kwargs)
        r['Setting'] = name
        results.append(r)

    df = pd.DataFrame(results)[['Setting','Precision','Recall','F1','ROC-AUC']]
    print("\n=== 실험 결과 비교표 ===")
    print(df.to_string(index=False))
    df.to_csv('data/raw/experiment_results.csv', index=False)
    print("\nSaved: data/raw/experiment_results.csv")
