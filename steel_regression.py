import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = os.path.join("data", "steelbench_core_open.csv")
OUT_DIR = "results"
TARGET = "tensile_strength"
FEATURES = ["C", "Mn", "Si", "Cr", "Ni", "Mo", "V", "Cu", "Al",
            "austenitize_T", "temper_T"]
RANDOM_STATE = 42


def load_data(path):
    # read raw csv, never modify the original file
    df = pd.read_csv(path)
    print("[1] raw data shape:", df.shape)
    print("[1] raw columns:", list(df.columns))
    return df


def inspect_data(df):
    print("\n[2] data inspection")
    print("sample count:", len(df))
    print("unique grade_id:", df["grade_id"].nunique())
    print("target unit: tensile_strength in MPa")
    print("\ndata_tier counts:")
    tier_counts = df["data_tier"].value_counts()
    for tier, n in tier_counts.items():
        print("  %s: %d" % (tier, n))
    measured = df[df["data_tier"].isin(["nims_measured", "kaggle_measured"])]
    spec = df[df["data_tier"] == "emk_spec_verified"]
    print("measured strength rows:", len(measured))
    print("specification-minimum rows:", len(spec))
    print("\nmissing values per column (raw):")
    print(df.isna().sum().to_string())
    print("\nexact duplicate rows:", int(df.duplicated().sum()))
    print("rows with missing target:", int(df[TARGET].isna().sum()))
    print("rows with missing grade_id:", int(df["grade_id"].isna().sum()))


def clean_data(df):
    print("\n[3] data cleaning")
    n0 = len(df)
    df = df.dropna(subset=[TARGET, "grade_id"]).copy()
    n1 = len(df)
    df = df[df[TARGET] > 0].copy()
    n2 = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    n3 = len(df)
    print("start rows:", n0)
    print("after dropping missing/invalid target and grade_id:", n1)
    print("after dropping non-positive target:", n2)
    print("after dropping exact duplicates:", n3)
    print("removed total:", n0 - n3)
    return df


def split_data(df):
    print("\n[4] grouped split by grade_id (GroupShuffleSplit)")
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(df, groups=df["grade_id"]))
    train = df.iloc[train_idx].copy()
    test = df.iloc[test_idx].copy()
    train_grades = set(train["grade_id"])
    test_grades = set(test["grade_id"])
    overlap = train_grades & test_grades
    print("train rows / test rows:", len(train), "/", len(test))
    print("train grades / test grades:", len(train_grades), "/", len(test_grades))
    print("overlapping grades:", len(overlap))
    assert len(overlap) == 0, "grade leakage between train and test"
    print("test data_tier counts:")
    print(test["data_tier"].value_counts().to_string())
    return train, test


def resolve_features(train):
    print("\n[5] feature availability check on training set")
    dropped = [c for c in FEATURES if train[c].isna().all()]
    if dropped:
        print("columns entirely missing in train, excluded:", dropped)
    else:
        print("columns entirely missing in train: none")
    used = [c for c in FEATURES if c not in dropped]
    print("features used:", used)
    return used


def build_models(features):
    linear = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", LinearRegression()),
    ])
    forest = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("model", RandomForestRegressor(n_estimators=100,
                                        random_state=RANDOM_STATE,
                                        n_jobs=-1)),
    ])
    return {"LinearRegression": linear, "RandomForest": forest}


def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    inspect_data(df)
    df = clean_data(df)
    train, test = split_data(df)
    features = resolve_features(train)

    print("\n[6] training models (test set never used for tuning)")
    models = build_models(features)
    fitted = {}
    for name, pipe in models.items():
        pipe.fit(train[features], train[TARGET])
        fitted[name] = pipe
        print("trained:", name)

    print("\n[7] test-set metrics")
    y_true = test[TARGET].to_numpy()
    preds = {name: pipe.predict(test[features]) for name, pipe in fitted.items()}

    rows = []
    for name, y_pred in preds.items():
        mae, rmse, r2 = evaluate(name, y_true, y_pred)
        rows.append({"model": name, "subset": "test_all",
                     "n_samples": len(test), "MAE_MPa": round(mae, 2),
                     "RMSE_MPa": round(rmse, 2), "R2": round(r2, 4)})
        print("%-16s MAE=%.2f  RMSE=%.2f  R2=%.4f" % (name, mae, rmse, r2))

        for tier in sorted(test["data_tier"].unique()):
            mask = (test["data_tier"] == tier).to_numpy()
            if mask.sum() < 2:
                continue
            m_mae, m_rmse, m_r2 = evaluate(name, y_true[mask], y_pred[mask])
            rows.append({"model": name, "subset": "test_tier:" + tier,
                         "n_samples": int(mask.sum()),
                         "MAE_MPa": round(m_mae, 2),
                         "RMSE_MPa": round(m_rmse, 2),
                         "R2": round(m_r2, 4)})
            print("  tier %-18s n=%4d MAE=%.2f RMSE=%.2f R2=%.4f"
                  % (tier, mask.sum(), m_mae, m_rmse, m_r2))

    metrics_df = pd.DataFrame(rows)
    metrics_path = os.path.join(OUT_DIR, "metrics.csv")
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print("\n[8] saved:", metrics_path)

    pred_df = pd.DataFrame({
        "heat_id": test["heat_id"].values,
        "source": test["source"].values,
        "data_tier": test["data_tier"].values,
        "tensile_strength_reference_MPa": test[TARGET].values,
    })
    for name, y_pred in preds.items():
        pred_df["predicted_" + name + "_MPa"] = np.round(y_pred, 2)
    pred_path = os.path.join(OUT_DIR, "predictions.csv")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    print("[8] saved:", pred_path, "rows:", len(pred_df))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (name, y_pred) in zip(axes, preds.items()):
        ax.scatter(y_true, y_pred, s=18, alpha=0.65, edgecolors="none")
        lo = float(min(y_true.min(), y_pred.min()))
        hi = float(max(y_true.max(), y_pred.max()))
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="y = x")
        ax.set_xlabel("Reference tensile strength (MPa)")
        ax.set_ylabel("Predicted tensile strength (MPa)")
        mae, rmse, r2 = evaluate(name, y_true, y_pred)
        ax.set_title("%s  MAE=%.1f  RMSE=%.1f  R2=%.3f" % (name, mae, rmse, r2))
        ax.legend(loc="upper left")
    fig.suptitle("Prediction vs reference on group-split test set")
    fig.tight_layout()
    cmp_path = os.path.join(OUT_DIR, "prediction_comparison.png")
    fig.savefig(cmp_path, dpi=150)
    plt.close(fig)
    print("[8] saved:", cmp_path)

    rf = fitted["RandomForest"].named_steps["model"]
    imputer = fitted["RandomForest"].named_steps["imputer"]
    out_names = list(imputer.get_feature_names_out(features))
    base_names = [n.replace("missingindicator_", "") for n in out_names]
    imp_all = pd.Series(rf.feature_importances_, index=base_names)
    imp = imp_all.groupby(level=0).sum().sort_values()
    fig, ax = plt.subplots(figsize=(8, 5.5))
    imp.plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Random forest feature importance")
    ax.set_title("Feature importance (RandomForest, 100 trees)")
    fig.tight_layout()
    imp_path = os.path.join(OUT_DIR, "feature_importance.png")
    fig.savefig(imp_path, dpi=150)
    plt.close(fig)
    print("[8] saved:", imp_path)

    print("\n[9] feature importance:")
    for k, v in imp.sort_values(ascending=False).items():
        print("  %-15s %.4f" % (k, v))


if __name__ == "__main__":
    main()
