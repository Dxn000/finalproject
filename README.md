# 💻 TECHCRAFT AI: COMPUTER HARDWARE & AI WORKSTATION PRICE PREDICTION

## 📥 Kaggle Dataset Download Link
- **Kaggle Dataset Name:** Laptop Price Dataset
- **Kaggle Dataset URL:** [https://www.kaggle.com/datasets/muhammetvarl/laptop-price](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
- **Local File in Repository:** `laptop_price.csv` (1,303 authentic computer hardware configurations).

---

## 🎯 Objective of the Application
The objective of this application is to estimate the retail market price (in Indian Rupees ₹) of personal computers, laptops, and AI development workstations based on hardware architectural specifications (Brand, Chassis Category, CPU family, GPU acceleration, RAM capacity, NVMe SSD storage, Display PPI resolution, and Operating System).

A Machine Learning pipeline combining automated feature scaling, one-hot encoding, and non-linear regression algorithms is trained on the Kaggle Hardware dataset and integrated with Streamlit to deliver an intuitive hardware spec-builder, market valuation, and upgrade return-on-investment (ROI) simulator.

---

## ❗ Problem Statement
Computer hardware configurations are non-linear, multi-attribute, and rapidly evolving. Consumers, software engineers, and AI researchers frequently face uncertainty when assessing whether a hardware configuration offers fair value for its compute capabilities. Furthermore, component upgrade costs (such as upgrading from 8GB to 32GB RAM for local LLMs or adding a dedicated NVIDIA GPU) vary widely across manufacturers.

The problem is to develop a machine learning application that learns the multivariate relationships between technical specifications and market prices, providing:
1. An accurate estimate of hardware retail pricing and monthly EMI financing in Indian Rupees (₹).
2. A display visual density calculation (PPI - Pixels Per Inch).
3. An AI & Deep Learning capability index (scoring systems for local LLM inference, CUDA training, and 3D rendering).
4. A component upgrade simulator demonstrating the financial delta and performance benefit of hardware changes.

---

## 🔍 Description of Inputs and Outputs

### Inputs (Hardware Architectural Parameters):
1. **Manufacturer / Brand** (`Company`): Dell, HP, Lenovo, Asus, Acer, MSI, Apple, Toshiba, etc.
2. **Chassis Category** (`TypeName`): Notebook, Gaming, Ultrabook, Workstation, 2-in-1 Convertible, Netbook.
3. **Processor Architecture** (`Cpu_Brand`): Intel Core i7, Intel Core i5, Intel Core i3, AMD Ryzen, Intel Other.
4. **Dedicated GPU** (`Gpu_Brand`): Nvidia (CUDA-enabled), AMD (Radeon), Intel (Integrated/Iris).
5. **System Memory** (`Ram_GB`): 4GB, 8GB, 12GB, 16GB, 24GB, 32GB, 64GB.
6. **Primary Storage** (`SSD_GB`): High-speed NVMe Solid State Drive (0GB to 2,048GB / 2TB).
7. **Secondary Storage** (`HDD_GB`): High-capacity mechanical drive (0GB to 2,048GB / 2TB).
8. **Display Screen Size** (`Inches`): Diagonal screen size (11.6" to 17.3").
9. **Display Resolution**: FHD (1920x1080), 2K QHD (2560x1440), 4K UHD (3840x2160), or HD (1366x768).
10. **Touchscreen & IPS Panel**: Display enhancement flags (Boolean).
11. **Chassis Weight** (`Weight_kg`): Physical weight in kilograms (0.9kg to 4.5kg).
12. **Operating System** (`OpSys_Category`): Windows 11, macOS, Linux, No OS.

### Outputs:
1. **Estimated Market Price:** Projected retail price in Indian Rupees (₹).
2. **No-Cost Monthly EMI:** 6-month and 12-month installment breakdowns (₹ / month).
3. **Display Density (PPI):** Precise pixels-per-inch sharpness calculation.
4. **AI & Deep Learning Readiness Score:** Performance index (0 to 100) and workstation capability tier.
5. **Hardware Upgrade Delta:** Real-time financial difference for memory or GPU component upgrades.

---

## 🧠 Explanation of How ML is Used

### Feature Engineering & Pipeline
1. The dataset `laptop_price.csv` contains 1,303 configurations.
2. Target price is converted to Indian Rupees (₹) and modeled on a log scale (`log1p`) to linearize exponential hardware pricing dynamics and guarantee non-negative predictions.
3. Continuous features (`Ram_GB`, `Weight_kg`, `PPI`, `SSD_GB`, `HDD_GB`) are normalized via `StandardScaler()`.
4. Categorical variables (`Company`, `TypeName`, `Cpu_Brand`, `Gpu_Brand`, `OpSys_Category`) are encoded using `OneHotEncoder(drop='first')` inside a `ColumnTransformer`.
5. Preprocessing and regression models are unified using a `scikit-learn` `Pipeline`.

### Model Evaluation & Comparison (85/15 Train-Test Split)
- **Multiple Linear Regression:**
  - $R^2$ Score: `0.8301` (83.0%)
  - Mean Absolute Error (MAE): `₹20,251.78`
  - Root Mean Squared Error (RMSE): `₹33,532.64`
- **Random Forest Regressor (Winner):**
  - $R^2$ Score: `0.8715` (87.2%)
  - Mean Absolute Error (MAE): `₹16,877.42`
  - Root Mean Squared Error (RMSE): `₹28,476.55`

The winning Random Forest pipeline is retrained on all observations and serialized as `laptop_price_model.pkl` using `joblib.dump()`.

---

## 🚀 Enhancements Over the Reference Project

| Dimension | Reference Project (Housing) | This Enhanced Project (Tech Hardware) |
|---|---|---|
| **Domain** | Single-variable housing price | Computer Hardware Engineering & AI Workstation Valuation |
| **Input Features** | Only 1 input variable (`area`) | **12 Multi-Modal Hardware Features** (CPU, GPU, RAM, NVMe, PPI, etc.) |
| **Feature Engineering** | None | PPI extraction, CPU/GPU categorization, dual SSD/HDD parsing, log transform |
| **Model Selection** | Simple Linear Regression ($R^2$ baseline) | Comparative benchmark: Linear Regression vs Random Forest Regressor ($R^2 = 87.2\%$) |
| **UI Experience** | Single input field and text output | **Multi-tab web app** with spec configurator, hardware upgrade simulator, and EDA charts |
| **Actionable Insights** | Estimated price only | Price in ₹, 12M EMI, Display PPI, AI Readiness Score (0-100), upgrade ROI |

---

## 💻 How to Run the Project Locally

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **(Optional) Train or Retrain the Model:**
   ```bash
   python train.py
   ```

3. **Launch the Streamlit Web Application:**
   ```bash
   streamlit run app.py
   ```
