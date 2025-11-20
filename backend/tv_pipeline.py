from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "SVM_RF_TV" / "Data" / "cleaned_tv_dataset_fi.csv"
MODEL_DIR = BASE_DIR / "SVM_RF_TV" / "Model"


NUMERIC_FEATURES = ["spec_Model_Year", "spec_Refresh_Rate", "spec_Screen_Size_Class"]
CAT_FEATURES = [
    "spec_Backlight_Type",
    "spec_Brand",
    "spec_Display_Type",
    "spec_ENERGY_STAR_Certified",
    "spec_High_Dynamic_Range_HDR",
    "spec_LED_Panel_Type",
    "spec_Remote_Control_Type",
    "spec_Resolution",
]

FEATURE_COLUMNS = CAT_FEATURES + NUMERIC_FEATURES

SEGMENT_BINS = [0, 250, 600, np.inf]
SEGMENT_LABELS = ["Entry", "Mid", "Premium"]


def _load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    for col in NUMERIC_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURE_COLUMNS + ["price"]).reset_index(drop=True)
    df["segment_label"] = pd.cut(
        df["price"], bins=SEGMENT_BINS, labels=SEGMENT_LABELS, include_lowest=True
    )
    return df


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ]
    )


def _train_segment_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    preprocess = _build_preprocessor()
    # Set probability=True to enable predict_proba for confidence scores
    classifier = SVC(kernel="rbf", C=100.0, gamma="scale", probability=True, random_state=42)
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("classifier", classifier),
        ]
    ).fit(X, y)


def _train_price_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    preprocess = _build_preprocessor()
    regressor = RandomForestRegressor(
        n_estimators=400, max_depth=16, min_samples_leaf=2, random_state=42
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", regressor),
        ]
    ).fit(X, y)


@dataclass
class TrainingReport:
    segment_accuracy: float
    segment_report: str
    price_mae: float
    price_r2: float


def train_and_save_models(
    data_path: Path = DATA_PATH, model_dir: Path = MODEL_DIR
) -> TrainingReport:
    df = _load_dataset(data_path)
    X = df[FEATURE_COLUMNS]
    y_segment = df["segment_label"]
    y_price = df["price"]

    X_train, X_test, y_seg_train, y_seg_test = train_test_split(
        X, y_segment, test_size=0.2, random_state=42, stratify=y_segment
    )
    X_price_train, X_price_test, y_price_train, y_price_test = train_test_split(
        X, y_price, test_size=0.2, random_state=42
    )

    segment_model = _train_segment_model(X_train, y_seg_train)
    price_model = _train_price_model(X_price_train, y_price_train)

    seg_predictions = segment_model.predict(X_test)
    segment_accuracy = accuracy_score(y_seg_test, seg_predictions)
    segment_report = classification_report(y_seg_test, seg_predictions)

    price_predictions = price_model.predict(X_price_test)
    price_mae = mean_absolute_error(y_price_test, price_predictions)
    price_r2 = r2_score(y_price_test, price_predictions)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(segment_model, model_dir / "segment_model.joblib")
    joblib.dump(price_model, model_dir / "price_model.joblib")

    return TrainingReport(
        segment_accuracy=segment_accuracy,
        segment_report=segment_report,
        price_mae=price_mae,
        price_r2=price_r2,
    )


class TVModelBundle:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.model_dir = model_dir
        # Priority 1: Try to load .joblib files first (newer, compatible models with probability=True)
        segment_file_joblib = model_dir / "segment_model.joblib"
        price_file_joblib = model_dir / "price_model.joblib"
        
        if segment_file_joblib.exists() and price_file_joblib.exists():
            try:
                self.segment_model: Pipeline = joblib.load(segment_file_joblib)
                self.price_model: Pipeline = joblib.load(price_file_joblib)
                logger.info(f"✅ Successfully loaded models from {segment_file_joblib} and {price_file_joblib}")
                # Check if model supports probability
                if hasattr(self.segment_model, "predict_proba"):
                    logger.info("✅ Model supports predict_proba for confidence scores")
                elif hasattr(self.segment_model, "decision_function"):
                    logger.info("✅ Model supports decision_function (will compute confidence from it)")
                else:
                    logger.warning("⚠️  Model does not support probability estimation")
                return  # Successfully loaded
            except (ValueError, TypeError, AttributeError) as e:
                logger.warning(f"Could not load .joblib models: {e}. Trying .pkl models...")
        
        # Priority 2: Try to load .pkl files (original models from notebook)
        segment_file_pkl = model_dir / "svm_best_model.pkl"
        price_file_pkl = model_dir / "rf_best_model.pkl"
        
        if segment_file_pkl.exists() and price_file_pkl.exists():
            try:
                self.segment_model: Pipeline = joblib.load(segment_file_pkl)
                self.price_model: Pipeline = joblib.load(price_file_pkl)
                logger.info(f"✅ Successfully loaded models from {segment_file_pkl} and {price_file_pkl}")
                # Check if model supports probability
                if hasattr(self.segment_model, "predict_proba"):
                    logger.info("✅ Model supports predict_proba for confidence scores")
                elif hasattr(self.segment_model, "decision_function"):
                    logger.info("✅ Model supports decision_function (will compute confidence from it)")
                else:
                    logger.warning("⚠️  Model does not support probability estimation - may have low confidence")
                return  # Successfully loaded
            except (ValueError, TypeError, AttributeError) as e:
                # Version incompatibility - models need to be retrained
                logger.warning(
                    f"Could not load .pkl models due to version incompatibility: {e}. "
                    f"Models will be retrained with current scikit-learn version."
                )
                # Fall through to retrain
        
        # If we get here, models couldn't be loaded - retrain them
        logger.info("Training new models compatible with current scikit-learn version...")
        train_and_save_models(model_dir=model_dir)
        
        # Now try loading the newly trained models
        segment_file = model_dir / "segment_model.joblib"
        price_file = model_dir / "price_model.joblib"
        if segment_file.exists() and price_file.exists():
            self.segment_model: Pipeline = joblib.load(segment_file)
            self.price_model: Pipeline = joblib.load(price_file)
        else:
            raise FileNotFoundError(
                f"Failed to train and load models from {model_dir}"
            )

    def predict(self, payload: Dict[str, str]) -> Dict[str, object]:
        input_df = pd.DataFrame([payload])
        
        # Predict segment
        segment_pred = self.segment_model.predict(input_df)[0]
        
        # Get confidence score - try predict_proba first, fallback to decision_function
        segment_proba = None
        try:
            if hasattr(self.segment_model, "predict_proba"):
                proba = self.segment_model.predict_proba(input_df)[0]
                segment_proba = proba.max()
                logger.info(f"📊 Confidence scores: {proba} → max={segment_proba:.4f} ({segment_proba*100:.2f}%)")
                # Log cả các class probabilities để debug
                if len(proba) == 3:  # Entry, Mid, Premium
                    logger.debug(f"   Entry: {proba[0]:.4f}, Mid: {proba[1]:.4f}, Premium: {proba[2]:.4f}")
            elif hasattr(self.segment_model, "decision_function"):
                # If no predict_proba, use decision_function and normalize
                decision = self.segment_model.decision_function(input_df)[0]
                logger.debug(f"Decision scores: {decision}")
                # Normalize decision scores to [0, 1] using softmax-like approach
                exp_scores = np.exp(decision - np.max(decision))
                proba = exp_scores / exp_scores.sum()
                segment_proba = proba.max()
                logger.info(f"📊 Confidence from decision_function: {segment_proba:.4f} ({segment_proba*100:.2f}%)")
            else:
                logger.warning("⚠️  Model does not support probability estimation")
        except Exception as e:
            logger.warning(f"⚠️  Could not compute confidence: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Predict price
        price_pred = float(self.price_model.predict(input_df)[0])
        suggested_price = round(price_pred, 2)
        
        confidence_str = f"{segment_proba:.4f}" if segment_proba else "N/A"
        logger.info(
            f"Prediction: segment={segment_pred}, confidence={confidence_str}, "
            f"price=${suggested_price}"
        )

        return {
            "segment": segment_pred,
            "segment_confidence": segment_proba,
            "suggested_price": suggested_price,
            "price_range": self._price_range(price_pred),
        }

    @staticmethod
    def _price_range(price: float, delta: float = 75.0) -> Tuple[float, float]:
        low = max(price - delta, 0)
        high = price + delta
        return round(low, 2), round(high, 2)


def get_feature_options(data_path: Path = DATA_PATH) -> Dict[str, List[str]]:
    # Load raw data để lấy tất cả options, không filter
    df_raw = pd.read_csv(data_path)
    # Convert numeric columns
    for col in NUMERIC_FEATURES:
        df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")
    
    # Get options từ raw data (trước khi dropna)
    options = {col: sorted(df_raw[col].dropna().unique().tolist()) for col in CAT_FEATURES}
    options.update(
        {
            "spec_Model_Year": sorted(df_raw["spec_Model_Year"].dropna().unique().tolist()),
            "spec_Refresh_Rate": sorted(df_raw["spec_Refresh_Rate"].dropna().unique().tolist()),
            "spec_Screen_Size_Class": sorted(df_raw["spec_Screen_Size_Class"].dropna().unique().tolist()),
        }
    )
    return options

