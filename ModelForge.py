"""
ModelForge - my end-to-end AutoML pipeline.
Give it a CSV/Excel file + target column, it cleans the data,
engineers features, trains a bunch of models and saves the best one.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, KFold
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.metrics import r2_score, accuracy_score
from xgboost import XGBRegressor, XGBClassifier

warnings.filterwarnings("ignore")


class ModelForge:

    def __init__(self, data_path, target_col):
        self.data_path = data_path
        self.target_col = target_col
        self.save_dir = os.path.dirname(data_path) or "."

        self.df = None
        self.task_type = None
        self.encoders = {}
        self.scaler = None
        self.final_features = []
        self.dropped_cols = []

        self.missing_drop_limit = 0.5
        self.low_variance_limit = 0.01
        self.corr_limit = 0.05
        self.outlier_zscore = 3.5

    @staticmethod
    def _is_text_column(series):
        return (
            series.dtype == "object"
            or isinstance(series.dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(series)
        )

    def load_data(self):
        ext = os.path.splitext(self.data_path)[1].lower()
        if ext == ".csv":
            self.df = pd.read_csv(self.data_path)
        elif ext in (".xls", ".xlsx"):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError("Only CSV or Excel files are supported right now.")

        print(f"\nLoaded dataset -> {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        self._detect_task_type()

    def _detect_task_type(self):
        target = self.df[self.target_col]
        n_unique = target.nunique()
        unique_ratio = n_unique / len(target)

        if self._is_text_column(target):
            self.task_type = "classification"
        elif n_unique <= 20 and unique_ratio <= 0.05:
            self.task_type = "classification"
        else:
            self.task_type = "regression"

        print(f"Detected task type: {self.task_type.upper()}  (target has {n_unique} unique values)")

    def clean_data(self):
        df = self.df.copy()

        # columns that are mostly empty aren't worth saving
        drop_these = []
        for col in df.columns:
            if col == self.target_col:
                continue
            missing_pct = df[col].isna().mean()
            if missing_pct > self.missing_drop_limit:
                drop_these.append(col)
                print(f"  Dropping '{col}' - {missing_pct:.0%} missing")

        df = df.drop(columns=drop_these)
        self.dropped_cols += drop_these

        for col in df.columns:
            if df[col].isna().sum() == 0:
                continue
            if self._is_text_column(df[col]):
                df[col] = df[col].fillna(df[col].mode()[0])
            else:
                if abs(df[col].skew()) > 1.0:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mean())

        before = len(df)
        df = df.dropna(subset=[self.target_col])
        if len(df) < before:
            print(f"  Dropped {before - len(df)} rows with missing target")

        for col in df.columns:
            if self._is_text_column(df[col]):
                enc = LabelEncoder()
                df[col] = enc.fit_transform(df[col].astype(str))
                self.encoders[col] = enc

        flat_cols = []
        for col in df.columns:
            if col == self.target_col:
                continue
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].var() < self.low_variance_limit:
                flat_cols.append(col)

        if flat_cols:
            print(f"  Dropping near-constant columns: {flat_cols}")
            df = df.drop(columns=flat_cols)
            self.dropped_cols += flat_cols

        self.df = df
        out_path = os.path.join(self.save_dir, "cleaned_data.csv")
        self.df.to_csv(out_path, index=False)
        print(f"Cleaned data saved -> {out_path}")

    def engineer_features(self):
        df = self.df.copy()
        numeric_cols = [c for c in df.columns if c != self.target_col and pd.api.types.is_numeric_dtype(df[c])]

        # clip outliers instead of dropping rows, don't want to lose data
        for col in numeric_cols:
            z = np.abs(stats.zscore(df[col]))
            if (z > self.outlier_zscore).any():
                low, high = df[col].quantile([0.01, 0.99])
                df[col] = df[col].clip(low, high)

        corr = df[numeric_cols + [self.target_col]].corr().abs()[self.target_col].drop(self.target_col)
        weak_features = corr[corr < self.corr_limit].index.tolist()
        if weak_features:
            print(f"  Dropping weakly-correlated features: {weak_features}")
            df = df.drop(columns=weak_features)
            self.dropped_cols += weak_features

        strong_numeric = [c for c in numeric_cols if c not in weak_features]

        for col in strong_numeric:
            if abs(df[col].skew()) > 0.75:
                df[col] = np.log1p(df[col] - df[col].min() + 1)

        self.scaler = StandardScaler()
        if strong_numeric:
            df[strong_numeric] = self.scaler.fit_transform(df[strong_numeric])

        self.final_features = [c for c in df.columns if c != self.target_col]
        self.df = df
        print(f"  Final feature set ({len(self.final_features)}): {self.final_features}")

    def _candidate_models(self):
        if self.task_type == "regression":
            return {
                "Linear Regression": (LinearRegression(), {}),
                "Ridge": (Ridge(), {"alpha": [0.1, 1.0, 10.0]}),
                "Lasso": (Lasso(max_iter=5000), {"alpha": [0.01, 0.1, 1.0]}),
                "Decision Tree": (DecisionTreeRegressor(random_state=42), {"max_depth": [None, 5, 10, 15]}),
                "Random Forest": (RandomForestRegressor(random_state=42), {"n_estimators": [100, 200], "max_depth": [None, 10]}),
                "Extra Trees": (ExtraTreesRegressor(random_state=42), {"n_estimators": [100, 200]}),
                "Gradient Boosting": (GradientBoostingRegressor(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}),
                "XGBoost": (XGBRegressor(random_state=42, verbosity=0), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]}),
                "KNN": (KNeighborsRegressor(), {"n_neighbors": [3, 5, 7]}),
                "SVR": (SVR(), {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}),
            }
        return {
            "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=42), {"C": [0.1, 1.0, 10.0]}),
            "Decision Tree": (DecisionTreeClassifier(random_state=42), {"max_depth": [None, 5, 10, 15]}),
            "Random Forest": (RandomForestClassifier(random_state=42), {"n_estimators": [100, 200], "max_depth": [None, 10]}),
            "Extra Trees": (ExtraTreesClassifier(random_state=42), {"n_estimators": [100, 200]}),
            "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]}),
            "XGBoost": (XGBClassifier(random_state=42, verbosity=0, eval_metric="logloss"), {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 6]}),
            "Naive Bayes": (GaussianNB(), {}),
            "SVM": (SVC(random_state=42, probability=True), {"C": [0.1, 1.0, 10.0], "kernel": ["rbf", "linear"]}),
            "KNN": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7]}),
        }

    def train_models(self):
        X = self.df[self.final_features]
        y = self.df[self.target_col]

        if self.task_type == "classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "accuracy"
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = "r2"

        models = self._candidate_models()

        print("\n" + "=" * 55)
        print(f"  {'CLASSIFICATION' if self.task_type == 'classification' else 'REGRESSION'} MODEL COMPARISON")
        print("=" * 55)
        print(f"  {'Model':<20}{'Test Score':>14}{'CV Mean':>14}")
        print("  " + "-" * 48)

        best_score = -float("inf")
        best_model = None
        best_name = None
        leaderboard = []

        for name, (model, param_grid) in models.items():
            try:
                grid = GridSearchCV(model, param_grid, cv=cv, scoring=scoring, n_jobs=-1)
                grid.fit(X_train, y_train)
                preds = grid.predict(X_test)

                if self.task_type == "regression":
                    test_score = r2_score(y_test, preds)
                else:
                    test_score = accuracy_score(y_test, preds)

                cv_score = grid.best_score_
                print(f"  {name:<20}{test_score*100:>13.1f}%{cv_score*100:>13.1f}%")
                leaderboard.append({"model": name, "test_score": test_score, "cv_mean": cv_score})

                if test_score > best_score:
                    best_score, best_model, best_name = test_score, grid.best_estimator_, name

            except Exception as e:
                print(f"  {name:<20}{'failed':>14}  ({e})")

        print("-" * 50)
        print(f"  Best model: {best_name}  ->  {best_score*100:.1f}%")

        artifacts = {
            "model": best_model,
            "scaler": self.scaler,
            "encoders": self.encoders,
            "features": self.final_features,
            "task_type": self.task_type,
            "target_col": self.target_col,
        }
        model_path = os.path.join(self.save_dir, "best_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(artifacts, f)

        print(f"  Saved best model -> {model_path}")
        return leaderboard


if __name__ == "__main__":
    path = input("Path to your dataset (csv/xlsx): ").strip()
    target = input("Name of the target column: ").strip()

    if not os.path.exists(path):
        print("Can't find that file, check the path again.")
    else:
        pipeline = ModelForge(path, target)
        pipeline.load_data()
        pipeline.clean_data()
        pipeline.engineer_features()
        pipeline.train_models()