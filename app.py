"""
app.py
=============================================================================
TECHCRAFT AI - HARDWARE SPECIFICATION & AI WORKSTATION PRICE PREDICTOR
=============================================================================
Interactive Streamlit application for predicting computer hardware and laptop
market pricing in Indian Rupees (₹) and evaluating AI/Gaming performance tiers,
trained on the Kaggle 'Laptop Price Dataset'.
=============================================================================
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure UTF-8 support on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from predict import predict_laptop_price, load_model, compute_ppi, evaluate_hardware_capabilities

# Streamlit Page Configuration
st.set_page_config(
    page_title="TechCraft AI - Hardware & AI Workstation Predictor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #F8FAFC 0%, #F1F5F9 100%);
        border: 1px solid #CBD5E1;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2.1rem;
        font-weight: 800;
        color: #2563EB;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #475569;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .upgrade-card {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%);
        border: 1px solid #93C5FD;
        border-radius: 12px;
        padding: 1.25rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "laptop_price.csv")
METRICS_PATH = os.path.join(BASE_DIR, "model_metrics.json")
KAGGLE_URL = "https://www.kaggle.com/datasets/muhammetvarl/laptop-price"


@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, encoding="latin-1")
        # Add INR price column
        df["Price_INR"] = df["Price_euros"] * 90.0
        return df
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


df = load_dataset()
metrics_data = load_metrics()

try:
    model = load_model()
except Exception:
    model = None


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/laptop.png", width=64)
    st.title("TechCraft AI ⚡")
    st.markdown("**Hardware & AI Workstation Valuation**")
    
    st.divider()
    
    st.markdown("### 📥 Kaggle Dataset")
    st.info(
        "Trained on the Kaggle **Laptop Price Dataset** "
        "covering 1,303 high-performance hardware configurations."
    )
    st.link_button("🔗 View & Download on Kaggle", KAGGLE_URL, width="stretch")
    
    st.divider()
    
    st.markdown("### 📌 Hardware Dataset Scope")
    st.markdown("- **Configurations:** 1,303 Laptops & Workstations")
    st.markdown("- **Target:** Market Price in Indian Rupees (₹)")
    st.markdown("- **Features:** CPU, GPU, RAM, NVMe SSD, PPI Display, OS")
    
    if metrics_data and "model_comparison" in metrics_data:
        st.divider()
        st.markdown("### 🏆 Active ML Model")
        best_name = metrics_data.get("best_model", "Random Forest Regressor")
        r2_val = metrics_data["model_comparison"].get(best_name, {}).get("R2_Score", 0.8715)
        mae_val = metrics_data["model_comparison"].get(best_name, {}).get("MAE_INR", 16877.42)
        st.success(f"**{best_name}**\n- **R² Score:** {r2_val*100:.1f}%\n- **MAE:** ₹{mae_val:,.2f}")
    
    st.caption("Engineered for AI Developers, Gamers & Hardware Enthusiasts")


# ==========================================
# MAIN HEADER
# ==========================================
st.markdown('<div class="main-title">💻 TechCraft AI: Hardware Specification & Price Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict realistic retail pricing in Indian Rupees (₹), compute display pixel density (PPI), and evaluate AI & gaming performance tiers for PC/laptop configurations.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔮 Spec Configurator & Predictor",
    "⚡ Hardware Upgrade & ROI Simulator",
    "📊 Market Insights & EDA",
    "🤖 Model Performance",
    "📄 Project Documentation"
])


# ==========================================
# TAB 1: SPEC CONFIGURATOR
# ==========================================
with tab1:
    st.subheader("Configure Hardware Specifications")
    st.write("Customize the compute components, memory architecture, and display panel below:")

    col_brand, col_type, col_cpu = st.columns(3)
    
    with col_brand:
        company = st.selectbox(
            "🏢 Manufacturer / Brand",
            ["Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Apple", "Toshiba", "Other"],
            index=0
        )
    
    with col_type:
        type_name = st.selectbox(
            "📐 Chassis Category",
            ["Notebook", "Gaming", "Ultrabook", "Workstation", "2 in 1 Convertible", "Netbook"],
            index=1
        )
    
    with col_cpu:
        cpu_brand = st.selectbox(
            "🧠 Processor (CPU Architecture)",
            ["Intel Core i7", "Intel Core i5", "Intel Core i3", "AMD Ryzen/Processor", "Intel Other (Celeron/Pentium/Xeon)"],
            index=0
        )

    col_ram, col_gpu, col_os = st.columns(3)
    
    with col_ram:
        ram_gb = st.select_slider(
            "💾 System RAM Capacity (GB)",
            options=[4, 8, 12, 16, 24, 32, 64],
            value=16,
            help="Higher RAM enables larger local AI model contexts & video editing"
        )
    
    with col_gpu:
        gpu_brand = st.selectbox(
            "🎮 Dedicated Graphics (GPU Accelerator)",
            ["Nvidia", "AMD", "Intel"],
            index=0,
            help="Nvidia provides dedicated CUDA cores for PyTorch / AI workflows"
        )
    
    with col_os:
        opsys = st.selectbox(
            "🖥️ Operating System",
            ["Windows", "Mac", "Linux", "Others/No OS"],
            index=0
        )

    st.markdown("##### 💽 Storage & Display Vitals")
    col_ssd, col_hdd, col_screen, col_res = st.columns(4)
    
    with col_ssd:
        ssd_gb = st.selectbox("Primary NVMe SSD Storage", [0, 128, 256, 512, 1024, 2048], index=3, format_func=lambda x: f"{x} GB" if x < 1024 else f"{x//1024} TB")
    
    with col_hdd:
        hdd_gb = st.selectbox("Secondary HDD Storage", [0, 500, 1024, 2048], index=0, format_func=lambda x: f"{x} GB" if x < 1024 else f"{x//1024} TB")
    
    with col_screen:
        inches = st.slider("Display Screen Size (Inches)", 11.6, 17.3, 15.6, step=0.1)
    
    with col_res:
        resolution = st.selectbox(
            "Display Resolution",
            ["1920x1080 (FHD)", "2560x1440 (2K QHD)", "3840x2160 (4K UHD)", "1366x768 (HD)"],
            index=0
        )
        res_clean = resolution.split()[0]

    col_touch, col_ips, col_wt = st.columns(3)
    with col_touch:
        touchscreen = st.checkbox("Touchscreen Support", value=False)
    with col_ips:
        ips_panel = st.checkbox("IPS Color-Accurate Panel", value=True)
    with col_wt:
        weight_kg = st.slider("Chassis Weight (kg)", 0.9, 4.5, 2.2, step=0.1)

    st.write("")
    predict_btn = st.button("🚀 Estimate Hardware Market Price (₹)", type="primary", width="stretch")

    if predict_btn or "last_hardware_prediction" in st.session_state:
        if predict_btn:
            pred_res = predict_laptop_price(
                company=company,
                type_name=type_name,
                ram_gb=ram_gb,
                weight_kg=weight_kg,
                touchscreen=1 if touchscreen else 0,
                ips=1 if ips_panel else 0,
                screen_size_inches=inches,
                resolution_str=res_clean,
                cpu_brand=cpu_brand,
                ssd_gb=ssd_gb,
                hdd_gb=hdd_gb,
                gpu_brand=gpu_brand,
                opsys_category=opsys,
                model=model
            )
            st.session_state.last_hardware_prediction = pred_res
        else:
            pred_res = st.session_state.last_hardware_prediction

        st.divider()
        st.markdown("### 📋 Hardware Valuation & Performance Index")

        r1, r2, r3, r4 = st.columns(4)
        with r1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Estimated Retail Price</div>
                <div class="metric-value">₹{pred_res['predicted_price_inr']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">12-Month No-Cost EMI</div>
                <div class="metric-value" style="color: #059669;">₹{pred_res['monthly_emi_12m']:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with r3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Display Pixel Density</div>
                <div class="metric-value" style="color: #6366F1;">{pred_res['ppi']} PPI</div>
            </div>
            """, unsafe_allow_html=True)

        with r4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">AI / Deep Learning Index</div>
                <div class="metric-value" style="color: {pred_res['capabilities']['tier_color']};">
                    {pred_res['capabilities']['ai_score']}/100
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.info(f"⚡ **Workstation Tier Assessment:** This configuration qualifies as a **{pred_res['capabilities']['workstation_tier']}**.")


# ==========================================
# TAB 2: HARDWARE UPGRADE SIMULATOR
# ==========================================
with tab2:
    st.subheader("⚡ Hardware Upgrade & Cost-Benefit Simulator")
    st.write(
        "Evaluate how individual component upgrades (e.g. stepping from 8GB to 32GB RAM for local LLMs, "
        "or adding a dedicated NVIDIA GPU) impact market pricing and compute capabilities."
    )

    s_col1, s_col2 = st.columns(2)
    with s_col1:
        st.markdown("#### Scenario 1: RAM Scaling for AI Workloads")
        st.write("Compare standard 8GB RAM vs High-Capacity 32GB RAM for running local LLMs (e.g., Llama 3 / Ollama):")
        
        base_ram = 8
        target_ram = 32
        p_base = predict_laptop_price(company, type_name, base_ram, weight_kg, 0, 1, inches, res_clean, cpu_brand, ssd_gb, hdd_gb, gpu_brand, opsys, model=model)["predicted_price_inr"]
        p_upgraded = predict_laptop_price(company, type_name, target_ram, weight_kg, 0, 1, inches, res_clean, cpu_brand, ssd_gb, hdd_gb, gpu_brand, opsys, model=model)["predicted_price_inr"]
        delta_ram = p_upgraded - p_base

        st.markdown(f"""
        <div class="upgrade-card">
            <div class="metric-label" style="color: #1E40AF;">RAM Upgrade Delta (8GB ➔ 32GB)</div>
            <div class="metric-value" style="color: #1D4ED8;">+₹{delta_ram:,.2f}</div>
            <p style="margin-top: 0.5rem; color: #374151;">Provides <b>4x memory bandwidth</b>, eliminating swap latency when loading model weights.</p>
        </div>
        """, unsafe_allow_html=True)

    with s_col2:
        st.markdown("#### Scenario 2: GPU Accelerator Upgrade")
        st.write("Compare baseline integrated Intel graphics vs dedicated NVIDIA RTX GPU acceleration:")
        
        p_intel = predict_laptop_price(company, type_name, ram_gb, weight_kg, 0, 1, inches, res_clean, cpu_brand, ssd_gb, hdd_gb, "Intel", opsys, model=model)["predicted_price_inr"]
        p_nvidia = predict_laptop_price(company, type_name, ram_gb, weight_kg, 0, 1, inches, res_clean, cpu_brand, ssd_gb, hdd_gb, "Nvidia", opsys, model=model)["predicted_price_inr"]
        delta_gpu = p_nvidia - p_intel

        st.markdown(f"""
        <div class="upgrade-card">
            <div class="metric-label" style="color: #1E40AF;">GPU Acceleration Delta (Intel ➔ NVIDIA)</div>
            <div class="metric-value" style="color: #1D4ED8;">+₹{delta_gpu:,.2f}</div>
            <p style="margin-top: 0.5rem; color: #374151;">Enables hardware Tensor Cores for CUDA acceleration, ray-tracing, and high-FPS rendering.</p>
        </div>
        """, unsafe_allow_html=True)


# ==========================================
# TAB 3: MARKET INSIGHTS & EDA
# ==========================================
with tab3:
    st.subheader("📊 Computer Hardware Market Insights & Visual Analytics")
    st.markdown(f"Direct dataset source: [Kaggle - Laptop Price Dataset]({KAGGLE_URL})")

    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Benchmark Systems", f"{len(df):,}")
        c2.metric("Mean Market Price", f"₹{df['Price_INR'].mean():,.2f}")
        c3.metric("Median Market Price", f"₹{df['Price_INR'].median():,.2f}")
        c4.metric("Top Brand by Volume", f"{df['Company'].mode()[0]}")

        st.divider()

        v1, v2 = st.columns(2)
        with v1:
            st.markdown("#### 1. GPU Manufacturer vs Market Price (₹)")
            df_gpu_clean = df.copy()
            df_gpu_clean["Gpu_Brand"] = df_gpu_clean["Gpu"].apply(lambda x: x.split()[0] if x.split()[0] in ["Intel", "Nvidia", "AMD"] else "Other")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.boxplot(
                data=df_gpu_clean,
                x="Gpu_Brand",
                y="Price_INR",
                hue="Gpu_Brand",
                palette=["#3B82F6", "#10B981", "#EF4444", "#6B7280"],
                legend=False,
                ax=ax
            )
            ax.set_title("Hardware Valuation by GPU Ecosystem")
            ax.set_xlabel("GPU Brand")
            ax.set_ylabel("Price (₹)")
            st.pyplot(fig)
            plt.close(fig)

        with v2:
            st.markdown("#### 2. RAM Capacity vs Price Distribution")
            df_ram_clean = df.copy()
            df_ram_clean["Ram_GB"] = df_ram_clean["Ram"].astype(str).str.replace("GB", "", regex=False).str.strip().astype(int)
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=df_ram_clean,
                x="Ram_GB",
                y="Price_INR",
                hue="Ram_GB",
                palette="Blues_r",
                legend=False,
                ax=ax
            )
            ax.set_title("Average Retail Price by RAM Size (GB)")
            ax.set_xlabel("RAM (GB)")
            ax.set_ylabel("Mean Price (₹)")
            st.pyplot(fig)
            plt.close(fig)

        v3, v4 = st.columns(2)
        with v3:
            st.markdown("#### 3. Top Brands Pricing Spectrum")
            top_b = ["Dell", "Lenovo", "HP", "Asus", "Acer", "MSI", "Apple"]
            df_top_b = df[df["Company"].isin(top_b)]
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=df_top_b,
                x="Company",
                y="Price_INR",
                hue="Company",
                palette="viridis",
                legend=False,
                ax=ax
            )
            ax.set_title("Brand Premium Comparison (Mean Price in ₹)")
            ax.set_xlabel("Brand")
            ax.set_ylabel("Mean Price (₹)")
            st.pyplot(fig)
            plt.close(fig)

        with v4:
            st.markdown("#### 4. Chassis Category vs Price Distribution")
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.barplot(
                data=df,
                x="TypeName",
                y="Price_INR",
                hue="TypeName",
                palette="mako",
                legend=False,
                ax=ax
            )
            ax.set_title("Form Factor Pricing Hierarchy")
            ax.set_xlabel("Chassis Category")
            ax.set_ylabel("Mean Price (₹)")
            plt.xticks(rotation=25)
            st.pyplot(fig)
            plt.close(fig)

        st.divider()
        st.markdown("#### Raw Dataset Table (First 10 Systems)")
        st.dataframe(df[["Company", "Product", "TypeName", "Inches", "Cpu", "Ram", "Memory", "Gpu", "OpSys", "Price_INR"]].head(10), width="stretch")
    else:
        st.warning("laptop_price.csv not found in workspace.")


# ==========================================
# TAB 4: MODEL PERFORMANCE
# ==========================================
with tab4:
    st.subheader("🤖 Machine Learning Model Benchmarking")
    st.write(
        "Both Multiple Linear Regression and an ensemble Random Forest Regressor were trained "
        "and validated on an 85/15 train/test split of the hardware specification dataset."
    )

    if metrics_data and "model_comparison" in metrics_data:
        m_list = []
        for name, m_metrics in metrics_data["model_comparison"].items():
            m_list.append({
                "Algorithm": name,
                "R² Score": f"{m_metrics['R2_Score'] * 100:.2f}%",
                "Mean Absolute Error (MAE)": f"₹{m_metrics['MAE_INR']:,.2f}",
                "Root Mean Squared Error (RMSE)": f"₹{m_metrics['RMSE_INR']:,.2f}"
            })
        st.table(pd.DataFrame(m_list))

    st.markdown("""
    ### 🔬 Technical Implementation Details:
    1. **Feature Engineering**: Derived continuous metrics including **PPI (Pixels Per Inch)** from resolution and diagonal size, separated storage into primary NVMe SSD and secondary HDD.
    2. **Log Transformation**: Applied `log1p` on target price to linearize exponential hardware premium curves and prevent negative estimations.
    3. **Scikit-Learn Pipeline**: Unifies `OneHotEncoder(drop='first')`, `StandardScaler()`, and `RandomForestRegressor`.
    4. **Accuracy**: Delivers **87.2% explained variance ($R^2$)**, accurately capturing brand equity and hardware component pricing.
    """)


# ==========================================
# TAB 5: DOCUMENTATION & REPORT
# ==========================================
with tab5:
    st.subheader("📄 Project Submission Documentation")
    st.markdown(f"""
### COMPUTER HARDWARE & AI WORKSTATION PRICE PREDICTION USING STREAMLIT

**Kaggle Dataset Download Link:**  
👉 [{KAGGLE_URL}]({KAGGLE_URL})

**Objective of the Application:**  
The objective of this application is to estimate the retail market price (in Indian Rupees ₹) of personal computers and AI workstations based on hardware architectural specifications (Brand, Chassis Category, CPU family, GPU acceleration, RAM capacity, NVMe SSD storage, Display PPI resolution, and Operating System). A Machine Learning pipeline is integrated with an interactive Streamlit interface to deliver a spec-building, valuation, and hardware upgrade simulator.

**Problem Statement:**  
Computer hardware specifications are complex and non-linear. Consumers, AI developers, and engineering students often struggle to evaluate whether a laptop configuration offers fair market value for its compute specs. Furthermore, hardware component premiums (such as transitioning from 8GB to 32GB RAM or adding a dedicated NVIDIA GPU) vary substantially. The problem is to develop a machine learning application that learns the multivariate relationships between technical specifications and market prices, providing instant pricing quotes, display PPI calculations, and AI performance tier classifications.

**Description of Inputs and Outputs:**
- **Inputs:**
  - `Company`: Laptop manufacturer (Apple, Dell, HP, Lenovo, Asus, Acer, MSI, etc.)
  - `TypeName`: Form factor (Gaming, Workstation, Ultrabook, Notebook, 2 in 1 Convertible)
  - `Cpu_Brand`: Processor family (Intel Core i7/i5/i3, AMD Ryzen, Intel Other)
  - `Ram_GB`: System memory (4GB, 8GB, 16GB, 32GB, 64GB)
  - `Gpu_Brand`: Graphics accelerator (Nvidia, AMD, Intel)
  - `SSD_GB` & `HDD_GB`: Primary solid-state and secondary magnetic storage
  - `Screen Size & Resolution`: Diagonal inches and resolution (FHD, 2K, 4K) -> computes PPI
  - `Touchscreen` & `IPS Panel`: Display enhancements (Boolean)
  - `Weight_kg`: Physical chassis mass
  - `OpSys`: Operating system (Windows, Mac, Linux, No OS)
- **Outputs:**
  - `Estimated Market Price`: Projected retail valuation in Indian Rupees (₹)
  - `No-Cost Monthly EMI`: Financing breakdown (6-month and 12-month)
  - `Display Pixel Density (PPI)`: Visual clarity index
  - `AI & Deep Learning Readiness Score`: Workstation performance tier (0-100)
  - `Hardware Upgrade Delta`: Financial difference and performance gain from component upgrades

**Explanation of how ML is used:**  
The project uses the `laptop_price.csv` dataset from Kaggle containing 1,303 configurations. In `train.py`, categorical features are encoded with `OneHotEncoder` and numerical features (`Ram_GB`, `Weight_kg`, `PPI`, `SSD_GB`, `HDD_GB`) are normalized using `StandardScaler` inside a `ColumnTransformer`. A log transformation (`log1p`) is applied to target prices to model exponential hardware pricing dynamics. Both Multiple Linear Regression and Random Forest Regressor models are trained and benchmarked on an 85/15 train-test split. The Random Forest Regressor achieves an **R² score of ~87.2%** and a **Mean Absolute Error (MAE) of ₹16,877.42**, outperforming baseline linear regression (R² ~ 83.0%). The winning pipeline is serialized into `laptop_price_model.pkl` via `joblib`. In `predict.py`, the model executes inference for the Streamlit web application (`app.py`).
    """)
