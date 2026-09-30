"""
train.py
=============================================================================
TECH HARDWARE & AI WORKSTATION PRICE PREDICTION - MODEL TRAINING SCRIPT
=============================================================================
This script loads the Kaggle 'Laptop Price Dataset' (laptop_price.csv),
performs feature engineering on hardware specifications (CPU, GPU, RAM,
NVMe SSD, PPI resolution, weight, and OS), trains and benchmarks Multiple
Linear Regression and Random Forest Regressor models, evaluates metrics
(R², MAE in ₹, RMSE in ₹), and serializes the winning pipeline to
'laptop_price_model.pkl' using joblib.
=============================================================================
"""

import os
import sys
import re
import json
import joblib
import numpy as np
import pandas as pd

# Ensure standard output safely encodes Unicode characters on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "laptop_price.csv")
MODEL_PATH = os.path.join(BASE_DIR, "laptop_price_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
INR_CONVERSION_RATE = 90.0  # 1 Euro ~ 90 INR


def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    
    # 1. Target: Price in INR
    if "Price_euros" in df.columns:
        df["Price_INR"] = df["Price_euros"] * INR_CONVERSION_RATE
    elif "Price" in df.columns:
        df["Price_INR"] = df["Price"] * INR_CONVERSION_RATE
    
    # 2. RAM: Extract numeric GB
    df["Ram_GB"] = df["Ram"].astype(str).str.replace("GB", "", regex=False).str.strip().astype(int)
    
    # 3. Weight: Extract numeric kg
    df["Weight_kg"] = df["Weight"].astype(str).str.replace("kg", "", regex=False).str.strip().astype(float)
    
    # 4. Display Features: Touchscreen & IPS Panel
    df["Touchscreen"] = df["ScreenResolution"].apply(lambda x: 1 if "Touchscreen" in str(x) else 0)
    df["IPS"] = df["ScreenResolution"].apply(lambda x: 1 if "IPS" in str(x) else 0)
    
    # 5. Pixels Per Inch (PPI)
    def parse_resolution(res_str):
        matches = re.findall(r"(\d{3,4})x(\d{3,4})", str(res_str))
        if matches:
            return int(matches[0][0]), int(matches[0][1])
        return 1920, 1080

    resolutions = df["ScreenResolution"].apply(parse_resolution)
    df["X_res"] = [r[0] for r in resolutions]
    df["Y_res"] = [r[1] for r in resolutions]
    df["PPI"] = (((df["X_res"] ** 2) + (df["Y_res"] ** 2)) ** 0.5 / df["Inches"]).round(2)
    
    # 6. CPU Categorization
    def categorize_cpu(cpu_str):
        cpu_str = str(cpu_str)
        if "Intel Core i7" in cpu_str:
            return "Intel Core i7"
        elif "Intel Core i5" in cpu_str:
            return "Intel Core i5"
        elif "Intel Core i3" in cpu_str:
            return "Intel Core i3"
        elif "AMD" in cpu_str:
            return "AMD Ryzen/Processor"
        elif "Intel" in cpu_str:
            return "Intel Other (Celeron/Pentium/Xeon)"
        else:
            return "Other"
            
    df["Cpu_Brand"] = df["Cpu"].apply(categorize_cpu)
    
    # 7. Storage Capacity (SSD & HDD in GB)
    def extract_storage(text):
        ssd = 0
        hdd = 0
        text = str(text)
        if "SSD" in text:
            match = re.search(r"(\d+)(GB|TB)\s*SSD", text)
            if match:
                val = int(match.group(1))
                unit = match.group(2)
                ssd = val * 1024 if unit == "TB" else val
        if "HDD" in text:
            match = re.search(r"(\d+)(GB|TB)\s*HDD", text)
            if match:
                val = int(match.group(1))
                unit = match.group(2)
                hdd = val * 1024 if unit == "TB" else val
        if ssd == 0 and hdd == 0:
            match = re.search(r"(\d+)(GB|TB)", text)
            if match:
                val = int(match.group(1))
                unit = match.group(2)
                ssd = val * 1024 if unit == "TB" else val
        return ssd, hdd

    storages = df["Memory"].apply(extract_storage)
    df["SSD_GB"] = [s[0] for s in storages]
    df["HDD_GB"] = [s[1] for s in storages]
    
    # 8. GPU Manufacturer
    def categorize_gpu(gpu_str):
        brand = str(gpu_str).split()[0]
        if brand in ["Intel", "Nvidia", "AMD"]:
            return brand
        return "Other"
        
    df["Gpu_Brand"] = df["Gpu"].apply(categorize_gpu)
    
    # 9. Operating System
    def categorize_os(os_str):
        os_str = str(os_str)
        if "Windows" in os_str:
            return "Windows"
        elif "Mac" in os_str or "macOS" in os_str:
            return "Mac"
        elif "Linux" in os_str:
            return "Linux"
        else:
            return "Others/No OS"
            
    df["OpSys_Category"] = df["OpSys"].apply(categorize_os)
    
    # 10. Top Manufacturers
    top_brands = ["Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Apple", "Toshiba"]
    df["Company"] = df["Company"].apply(lambda x: x if x in top_brands else "Other")
    
    return df


def train_and_evaluate():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please provide laptop_price.csv.")
    
    raw_df = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = extract_features(raw_df)
    print(f"Loaded and engineered features for {len(df)} computer hardware configurations.")
    
    feature_cols = [
        "Company", "TypeName", "Ram_GB", "Weight_kg", "Touchscreen",
        "IPS", "PPI", "Cpu_Brand", "SSD_GB", "HDD_GB", "Gpu_Brand", "OpSys_Category"
    ]
    
    X = df[feature_cols]
    y = np.log1p(df["Price_INR"])  # Log transform ensures non-negative, high-accuracy prediction
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    
    categorical_cols = ["Company", "TypeName", "Cpu_Brand", "Gpu_Brand", "OpSys_Category"]
    numerical_cols = ["Ram_GB", "Weight_kg", "PPI", "SSD_GB", "HDD_GB"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numerical_cols)
        ],
        remainder="passthrough"
    )
    
    # Benchmarking Models
    lr_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])
    
    rf_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=150, max_depth=14, random_state=42))
    ])
    
    models = {
        "Multiple Linear Regression": lr_pipeline,
        "Random Forest Regressor": rf_pipeline
    }
    
    results = {}
    print("\n" + "=" * 65)
    print(" COMPUTER HARDWARE SPECIFICATION BENCHMARKING (TEST SET) ")
    print("=" * 65)
    
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics on the real INR scale
        actual_inr = np.expm1(y_test)
        pred_inr = np.expm1(y_pred)
        
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(actual_inr, pred_inr)
        rmse = root_mean_squared_error(actual_inr, pred_inr)
        
        results[name] = {
            "R2_Score": round(float(r2), 4),
            "MAE_INR": round(float(mae), 2),
            "RMSE_INR": round(float(rmse), 2)
        }
        
        print(f"[{name}]")
        print(f"  -> R² Score: {r2:.4f} ({r2*100:.2f}%)")
        print(f"  -> Mean Absolute Error (MAE): ₹{mae:,.2f}")
        print(f"  -> Root Mean Squared Error (RMSE): ₹{rmse:,.2f}\n")
    
    best_model_name = max(results, key=lambda k: results[k]["R2_Score"])
    print(f"Best Performing Architecture: {best_model_name} (R² = {results[best_model_name]['R2_Score']})")
    
    # Retrain best pipeline on complete dataset
    final_pipeline = models[best_model_name]
    final_pipeline.fit(X, y)
    
    # Save model pipeline
    joblib.dump(final_pipeline, MODEL_PATH)
    print(f"Model saved to: {MODEL_PATH}")
    
    # Save metadata
    metadata = {
        "dataset_summary": {
            "total_records": len(df),
            "mean_price_inr": float(df["Price_INR"].mean()),
            "median_price_inr": float(df["Price_INR"].median()),
            "min_price_inr": float(df["Price_INR"].min()),
            "max_price_inr": float(df["Price_INR"].max()),
        },
        "model_comparison": results,
        "best_model": best_model_name
    }
    
    with open(METRICS_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Evaluation metrics saved to: {METRICS_PATH}")
    print("=" * 65 + "\n")
    
    return final_pipeline, results


if __name__ == "__main__":
    train_and_evaluate()
