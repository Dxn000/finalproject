"""
predict.py
=============================================================================
TECH HARDWARE & AI WORKSTATION PRICE PREDICTION - INFERENCE MODULE
=============================================================================
This module loads the trained scikit-learn pipeline from
'laptop_price_model.pkl' and exposes functions to predict computer hardware
pricing in Indian Rupees (₹) and estimate AI/Gaming compute tiers.
=============================================================================
"""

import os
import sys
import re
import joblib
import numpy as np
import pandas as pd

# Ensure standard output safely encodes Unicode characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "laptop_price_model.pkl")

_CACHED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    global _CACHED_MODEL
    if _CACHED_MODEL is not None:
        return _CACHED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. "
            "Please run 'python train.py' first to train and generate the model."
        )

    _CACHED_MODEL = joblib.load(model_path)
    return _CACHED_MODEL


def compute_ppi(inches: float, resolution_str: str) -> float:
    matches = re.findall(r"(\d{3,4})x(\d{3,4})", str(resolution_str))
    if matches:
        x_res, y_res = int(matches[0][0]), int(matches[0][1])
    else:
        x_res, y_res = 1920, 1080
    
    inches = max(inches, 10.0)
    ppi = (((x_res ** 2) + (y_res ** 2)) ** 0.5) / inches
    return round(float(ppi), 2)


def evaluate_hardware_capabilities(ram_gb: int, ssd_gb: int, cpu_brand: str, gpu_brand: str) -> dict:
    """
    Computes technical performance indices for AI/ML workloads and 3D Gaming/Rendering.
    """
    # AI / Deep Learning Index (0 - 100)
    ai_score = 20.0
    if gpu_brand == "Nvidia":
        ai_score += 45.0  # CUDA core acceleration for PyTorch / TensorFlow
    elif gpu_brand == "AMD":
        ai_score += 20.0  # ROCm support
    elif gpu_brand == "Intel":
        ai_score += 10.0

    if ram_gb >= 32:
        ai_score += 25.0
    elif ram_gb >= 16:
        ai_score += 18.0
    elif ram_gb >= 8:
        ai_score += 10.0

    if "i7" in cpu_brand or "i9" in cpu_brand or "Ryzen" in cpu_brand:
        ai_score += 10.0

    ai_score = min(round(ai_score, 1), 100.0)

    # Performance Classification
    if ai_score >= 80.0:
        workstation_tier = "Flagship AI & HPC Workstation"
        tier_color = "#8B5CF6"
    elif ai_score >= 60.0:
        workstation_tier = "Pro Developer & DL Training Rig"
        tier_color = "#3B82F6"
    elif ai_score >= 40.0:
        workstation_tier = "Mainstream Software Dev & Multitasking"
        tier_color = "#10B981"
    else:
        workstation_tier = "Entry-Level / Daily Computing"
        tier_color = "#6B7280"

    return {
        "ai_score": ai_score,
        "workstation_tier": workstation_tier,
        "tier_color": tier_color
    }


def predict_laptop_price(
    company: str,
    type_name: str,
    ram_gb: int,
    weight_kg: float,
    touchscreen: int,
    ips: int,
    screen_size_inches: float,
    resolution_str: str,
    cpu_brand: str,
    ssd_gb: int,
    hdd_gb: int,
    gpu_brand: str,
    opsys_category: str,
    model=None
) -> dict:
    if model is None:
        model = load_model()

    ppi = compute_ppi(screen_size_inches, resolution_str)

    input_df = pd.DataFrame([{
        "Company": str(company),
        "TypeName": str(type_name),
        "Ram_GB": int(ram_gb),
        "Weight_kg": float(weight_kg),
        "Touchscreen": int(touchscreen),
        "IPS": int(ips),
        "PPI": float(ppi),
        "Cpu_Brand": str(cpu_brand),
        "SSD_GB": int(ssd_gb),
        "HDD_GB": int(hdd_gb),
        "Gpu_Brand": str(gpu_brand),
        "OpSys_Category": str(opsys_category)
    }])

    # Model predicted on log1p scale -> convert back via expm1
    log_pred = float(model.predict(input_df)[0])
    predicted_inr = float(np.expm1(log_pred))
    
    # Boundary guards
    predicted_inr = max(round(predicted_inr, 2), 18000.0)

    # EMI calculation (12 months no-cost baseline)
    monthly_emi_12m = predicted_inr / 12.0
    monthly_emi_6m = predicted_inr / 6.0

    capabilities = evaluate_hardware_capabilities(ram_gb, ssd_gb, cpu_brand, gpu_brand)

    return {
        "predicted_price_inr": predicted_inr,
        "monthly_emi_12m": round(monthly_emi_12m, 2),
        "monthly_emi_6m": round(monthly_emi_6m, 2),
        "ppi": ppi,
        "capabilities": capabilities,
        "features": {
            "Company": company,
            "TypeName": type_name,
            "Ram_GB": ram_gb,
            "Cpu_Brand": cpu_brand,
            "Gpu_Brand": gpu_brand,
            "SSD_GB": ssd_gb,
            "HDD_GB": hdd_gb,
            "Screen": f"{screen_size_inches}\" ({resolution_str})"
        }
    }


if __name__ == "__main__":
    sample_pred = predict_laptop_price(
        company="Dell",
        type_name="Gaming",
        ram_gb=16,
        weight_kg=2.4,
        touchscreen=0,
        ips=1,
        screen_size_inches=15.6,
        resolution_str="1920x1080",
        cpu_brand="Intel Core i7",
        ssd_gb=512,
        hdd_gb=0,
        gpu_brand="Nvidia",
        opsys_category="Windows"
    )
    print("--- Sample Hardware Specification Prediction ---")
    print(f"Predicted Price: ₹{sample_pred['predicted_price_inr']:,.2f}")
    print(f"12M EMI        : ₹{sample_pred['monthly_emi_12m']:,.2f} / month")
    print(f"Display PPI    : {sample_pred['ppi']}")
    print(f"AI Tier        : {sample_pred['capabilities']['workstation_tier']} (Score: {sample_pred['capabilities']['ai_score']}/100)")
