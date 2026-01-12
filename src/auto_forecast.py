import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ElasticNet, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


@dataclass
class ForecastResult:
    latest_date: pd.Timestamp
    selected_model: str
    predicted_next_dq_score: float
    naive_predicted_next_dq_score: float
    metrics_summary: pd.DataFrame
    comparison_table: Optional[pd.DataFrame]
    artifact_path: Optional[Path]


def _drop_future_looking_columns(df: pd.DataFrame) -> pd.DataFrame:
    future_cols = [c for c in df.columns if "to_next" in c or c == "days_to_next_measurement"]
    if future_cols:
        return df.drop(columns=future_cols)
    return df


def _prepare_forecast_frame(
    df: pd.DataFrame,
    target_col: str = "DQ_SCORE",
    index_col: str = "calendar_date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.index.name != index_col:
        raise ValueError(f"Expected index name '{index_col}', got '{df.index.name}'")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")

    df = df.sort_index()
    df = _drop_future_looking_columns(df)

    df = df.copy()
    df["DQ_SCORE_next"] = df[target_col].shift(-1)

    df_labeled = df.dropna(subset=["DQ_SCORE_next"]).copy()
    df_latest = df.iloc[[-1]].copy()
    return df_labeled, df_latest


def _select_numeric_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include=[np.number])


def _build_feature_matrix(
    df_labeled: pd.DataFrame,
    drop_columns: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    drop_columns = drop_columns or []

    base_drop = ["DQ_SCORE_next"] + drop_columns
    X = df_labeled.drop(columns=[c for c in base_drop if c in df_labeled.columns], errors="ignore")
    X = _select_numeric_features(X)
    y = df_labeled["DQ_SCORE_next"].astype(float)

    if X.empty:
        raise ValueError("No numeric features available for forecasting after filtering")

    return X, y


def _candidate_models(random_state: int = 42) -> Dict[str, object]:
    return {
        "DecisionTree(depth=2)": DecisionTreeRegressor(max_depth=2, min_samples_leaf=2, random_state=random_state),
        "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state, max_iter=10000),
        "BayesianRidge": BayesianRidge(),
        "Ridge": Ridge(alpha=5.0, random_state=random_state),
    }


def _is_tree_model(name: str) -> bool:
    return name.startswith("DecisionTree")


def walk_forward_rmse(
    X: pd.DataFrame,
    y: pd.Series,
    models: Dict[str, object],
    min_train: Optional[int] = None,
    include_comparison_table: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    n = len(X)
    if n < 10:
        raise ValueError(f"Not enough labeled samples for walk-forward evaluation: {n}")

    min_train = min_train if min_train is not None else max(8, int(n * 0.4))
    if min_train >= n:
        raise ValueError(f"min_train ({min_train}) must be < number of samples ({n})")

    sq_err_by_model: Dict[str, List[float]] = {name: [] for name in models.keys()}
    sq_err_by_model["Naive(last_value)"] = []

    rows = [] if include_comparison_table else None

    for t in range(min_train, n):
        X_train = X.iloc[:t]
        y_train = y.iloc[:t]

        X_test = X.iloc[[t]]
        y_true = float(y.iloc[t])

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        current_val = float(X_test["DQ_SCORE"].iloc[0]) if "DQ_SCORE" in X_test.columns else float("nan")

        naive_pred = current_val if not np.isnan(current_val) else float(y_train.iloc[-1])
        sq_err_by_model["Naive(last_value)"].append((naive_pred - y_true) ** 2)

        row = None
        if include_comparison_table:
            row = {
                "date": X_test.index[0],
                "actual_DQ_SCORE_next": y_true,
                "pred_Naive(last_value)": naive_pred,
                "error_Naive(last_value)": naive_pred - y_true,
            }

        for name, model in models.items():
            if _is_tree_model(name):
                model.fit(X_train, y_train)
                pred = float(model.predict(X_test)[0])
            else:
                model.fit(X_train_scaled, y_train)
                pred = float(model.predict(X_test_scaled)[0])

            sq_err_by_model[name].append((pred - y_true) ** 2)

            if include_comparison_table and row is not None:
                row[f"pred_{name}"] = pred
                row[f"error_{name}"] = pred - y_true

        if include_comparison_table and row is not None and rows is not None:
            rows.append(row)

    rmse_table = []
    for name, sq_errs in sq_err_by_model.items():
        rmse = float(np.sqrt(np.mean(sq_errs))) if sq_errs else float("inf")
        rmse_table.append({"model": name, "RMSE": rmse})

    metrics_df = pd.DataFrame(rmse_table).sort_values(by="RMSE", ascending=True).reset_index(drop=True)

    comparison_df = None
    if include_comparison_table and rows is not None:
        comparison_df = pd.DataFrame(rows)

    return metrics_df, comparison_df


def train_and_forecast_next(
    data_path: Union[str, Path],
    output_models_dir: Union[str, Path] = "../models",
    index_col: str = "calendar_date",
    target_col: str = "DQ_SCORE",
    random_state: int = 42,
    min_train: Optional[int] = None,
    mode: str = "post_run",
    save_artifact: bool = True,
    include_comparison_table: bool = True,
) -> ForecastResult:
    data_path = Path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path, index_col=index_col, parse_dates=True)

    df_labeled, df_latest = _prepare_forecast_frame(df, target_col=target_col, index_col=index_col)

    drop_cols = ["TRUST_SCORE_CATEGORY", "DQ_SCORE_CATEGORY"]

    if mode not in {"post_run", "pre_run"}:
        raise ValueError("mode must be 'post_run' or 'pre_run'")

    if mode == "pre_run":
        outcome_like = [
            c
            for c in df_labeled.columns
            if c.startswith("daily_")
            or "violations" in c
            or "failed" in c
            or "error" in c
            or "skipped" in c
        ]
        drop_cols.extend(outcome_like)

    X, y = _build_feature_matrix(df_labeled, drop_columns=drop_cols)

    models = _candidate_models(random_state=random_state)
    metrics_summary, comparison_table = walk_forward_rmse(
        X=X,
        y=y,
        models=models,
        min_train=min_train,
        include_comparison_table=include_comparison_table,
    )

    best_model_name = str(metrics_summary.iloc[0]["model"])

    latest_date = df_latest.index[0]
    X_latest = df_latest.reindex(columns=X.columns, fill_value=0.0)
    X_latest = _select_numeric_features(X_latest)

    naive_pred = float(df_latest[target_col].iloc[0])

    artifact_path = None

    if best_model_name == "Naive(last_value)":
        selected_pred = naive_pred
        selected_model = None
        scaler = None
    else:
        selected_model = models[best_model_name]
        scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X)
        X_latest_scaled = scaler.transform(X_latest)

        if _is_tree_model(best_model_name):
            selected_model.fit(X, y)
            selected_pred = float(selected_model.predict(X_latest)[0])
        else:
            selected_model.fit(X_scaled, y)
            selected_pred = float(selected_model.predict(X_latest_scaled)[0])

    if save_artifact:
        output_models_dir = Path(output_models_dir)
        output_models_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        artifact_path = output_models_dir / f"dq_score_next_forecaster_{stamp}.pkl"

        artifact = {
            "created_utc": stamp,
            "data_path": str(data_path),
            "mode": mode,
            "index_col": index_col,
            "target_col": target_col,
            "selected_model": best_model_name,
            "feature_columns": list(X.columns),
            "metrics_summary": metrics_summary,
            "comparison_table": comparison_table,
            "model_object": selected_model,
            "scaler": scaler,
        }

        joblib.dump(artifact, artifact_path)

    return ForecastResult(
        latest_date=latest_date,
        selected_model=best_model_name,
        predicted_next_dq_score=float(selected_pred),
        naive_predicted_next_dq_score=float(naive_pred),
        metrics_summary=metrics_summary,
        comparison_table=comparison_table,
        artifact_path=artifact_path,
    )
