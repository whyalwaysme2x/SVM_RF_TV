from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, validator

from tv_pipeline import (
    CAT_FEATURES,
    MODEL_DIR,
    NUMERIC_FEATURES,
    SEGMENT_LABELS,
    TVModelBundle,
    get_feature_options,
    train_and_save_models,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="TV Segment & Price Advisor",
    version="0.1.0",
    description=(
        "Serve SVM + RandomForest models to predict TV market segment and suggested price."
        " Designed to plug into a storefront-like UI."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (frontend)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

model_bundle: TVModelBundle | None = None
feature_options_cache: Dict[str, list] | None = None


class TVPayload(BaseModel):
    spec_Backlight_Type: str = Field(..., description="Backlight technology, e.g. Edge-lit")
    spec_Brand: str
    spec_Display_Type: str
    spec_ENERGY_STAR_Certified: str
    spec_High_Dynamic_Range_HDR: str
    spec_LED_Panel_Type: str
    spec_Remote_Control_Type: str
    spec_Resolution: str
    spec_Model_Year: int
    spec_Refresh_Rate: int
    spec_Screen_Size_Class: float

    @validator(
        "spec_Backlight_Type",
        "spec_Brand",
        "spec_Display_Type",
        "spec_ENERGY_STAR_Certified",
        "spec_High_Dynamic_Range_HDR",
        "spec_LED_Panel_Type",
        "spec_Remote_Control_Type",
        "spec_Resolution",
        pre=True,
    )
    def strip_text(cls, value: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Field cannot be empty")
        return value.strip()

    @validator("spec_Model_Year", "spec_Refresh_Rate", pre=True)
    def coerce_int(cls, value):
        if value in ("", None):
            raise ValueError("Numeric field required")
        return int(float(value))

    @validator("spec_Screen_Size_Class", pre=True)
    def coerce_float(cls, value):
        if value in ("", None):
            raise ValueError("Screen size required")
        return float(value)


def _ensure_models(model_dir: Path = MODEL_DIR) -> None:
    # Check for existing .pkl models first
    segment_file_pkl = model_dir / "svm_best_model.pkl"
    price_file_pkl = model_dir / "rf_best_model.pkl"
    
    # Check for .joblib models
    segment_file_joblib = model_dir / "segment_model.joblib"
    price_file_joblib = model_dir / "price_model.joblib"
    
    # If models exist, TVModelBundle will try to load them
    # If they fail due to version incompatibility, it will retrain automatically
    if (segment_file_pkl.exists() and price_file_pkl.exists()) or \
       (segment_file_joblib.exists() and price_file_joblib.exists()):
        return
    
    logger.info("Model artifacts not found. Training from scratch...")
    train_and_save_models(model_dir=model_dir)
    logger.info("Training completed.")


@app.on_event("startup")
def _startup() -> None:
    global model_bundle, feature_options_cache
    _ensure_models()
    model_bundle = TVModelBundle()
    feature_options_cache = get_feature_options()


@app.get("/")
def root():
    """Serve frontend index.html"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "TV Segment & Price Advisor API",
        "version": "0.1.0",
        "endpoints": {
            "health": "/api/health",
            "options": "/api/options",
            "predict": "/api/predict",
            "docs": "/docs",
        },
        "note": "Frontend not found. Please ensure frontend/index.html exists.",
    }


@app.get("/api/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/options")
def get_options() -> Dict[str, Any]:
    if feature_options_cache is None:
        raise HTTPException(status_code=503, detail="Options not ready yet.")
    return {"features": feature_options_cache, "segments": SEGMENT_LABELS}


@app.get("/api/model-info")
def get_model_info() -> Dict[str, Any]:
    """Get information about loaded model"""
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    
    info = {
        "model_dir": str(model_bundle.model_dir),
        "has_predict_proba": hasattr(model_bundle.segment_model, "predict_proba"),
        "has_decision_function": hasattr(model_bundle.segment_model, "decision_function"),
        "model_type": type(model_bundle.segment_model).__name__,
    }
    
    # Check which model files exist
    segment_joblib = model_bundle.model_dir / "segment_model.joblib"
    price_joblib = model_bundle.model_dir / "price_model.joblib"
    segment_pkl = model_bundle.model_dir / "svm_best_model.pkl"
    price_pkl = model_bundle.model_dir / "rf_best_model.pkl"
    
    info["available_models"] = {
        "segment_model.joblib": segment_joblib.exists(),
        "price_model.joblib": price_joblib.exists(),
        "svm_best_model.pkl": segment_pkl.exists(),
        "rf_best_model.pkl": price_pkl.exists(),
    }
    
    return info


@app.post("/api/predict")
def predict(payload: TVPayload) -> Dict[str, Dict[str, object]]:
    if model_bundle is None:
        raise HTTPException(status_code=503, detail="Model not ready yet.")
    try:
        result = model_bundle.predict(payload.dict())
        # Log confidence để debug
        if result.get("segment_confidence"):
            conf_pct = result["segment_confidence"] * 100
            logger.info(f"API Prediction: segment={result['segment']}, confidence={conf_pct:.2f}%, price=${result['suggested_price']}")
            if conf_pct < 50:
                logger.warning(f"⚠️  Low confidence detected: {conf_pct:.2f}%")
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"input": payload.dict(), "prediction": result}

