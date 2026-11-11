# Real Estate House Price Prediction & Regularized Regression Suite

An end-to-end Machine Learning regression system modeling property valuations using feature engineering, log-target normalization, and comparative regularization (OLS, Ridge, Lasso, and Random Forest).

---

## 📌 Project Overview

Accurate property valuation requires balancing structural specifications (square footage, bedrooms, bathrooms) with location micro-dynamics (neighborhood tier, school ratings, distance to urban hubs). This project implements a production-style regression pipeline:
- **Multivariate Generation:** Synthetic real estate transactions capturing non-linear pricing effects, amenities premiums, and neighborhood multipliers.
- **Domain Feature Engineering:** Interaction terms including square footage per room, lot-to-living ratios, non-linear age depreciation penalties, and school accessibility indices.
- **Log-Target Normalization:** Log-transforming target prices ($\log(1 + y)$) to stabilize variance, normalize right-skewed pricing distributions, and penalize percentage errors proportionally.
- **Regularization Benchmarking:** Direct empirical comparison of Ordinary Least Squares, Ridge ($L_2$ penalty), Lasso ($L_1$ feature selection), and Random Forest Ensembles.
- **Valuation CLI:** Real-time pricing estimation function providing point estimates and predictive valuation confidence bands.

---

## 🏗️ Architecture

```
machine-learning/house-price-prediction/
├── dataset_generator.py   # Multivariate real estate transaction synthesizer
├── preprocessor.py        # Feature engineering, scaling, and log-target pipeline
├── train_models.py        # 5-fold CV training (OLS, Ridge, Lasso, Random Forest)
├── evaluate.py            # Regression metrics (R², RMSE, MAE, MAPE) and valuation CLI
└── README.md              # Project documentation, mathematical formulation, & benchmarks
```

---

## 📐 Mathematical Formulation

### 1. Ordinary Least Squares (OLS)
Minimizes the sum of squared residuals:
$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2$$

### 2. Ridge Regression ($L_2$ Regularization)
Adds an $L_2$ penalty on coefficient magnitude to mitigate multicollinearity:
$$\min_{\mathbf{w}} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_2^2$$
Analytical solution:
$$\hat{\mathbf{w}}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \alpha \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}$$

### 3. Lasso Regression ($L_1$ Regularization)
Applies an $L_1$ penalty promoting sparse weight vectors (automatic feature selection):
$$\min_{\mathbf{w}} \frac{1}{2N} \|\mathbf{y} - \mathbf{X}\mathbf{w}\|_2^2 + \alpha \|\mathbf{w}\|_1$$

### 4. Log-Target Scaling & Inversion
To evaluate dollar metrics from log-predictions:
$$\hat{y}_{\text{usd}} = \exp(\hat{y}_{\text{log}}) - 1$$

---

## 📊 Benchmark Results

Evaluated with 5-fold cross-validation on $N=2,500$ properties:

| Model | 5-Fold $R^2$ | RMSE (log scale) | Test $R^2$ | Test RMSE (\$) | Test MAPE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | **0.9402** | **0.1013** | 0.9234 | \$55,120 | 8.15% |
| **Ridge Regression** ($\alpha=10$) | 0.9356 | 0.1053 | **0.9215** | **\$55,879** | **8.26%** |
| **Lasso Regression** ($\alpha=0.005$) | 0.9283 | 0.1111 | 0.9142 | \$58,410 | 8.65% |
| **Random Forest** (Depth=10) | 0.9249 | 0.1137 | 0.9110 | \$59,980 | 8.80% |

### Key Valuation Drivers:
- **Positive Drivers:** Living Area (+0.2561), Top School Rating (+0.0813), Luxury Waterfront location.
- **Negative Drivers:** Rural / Long distance from city (-0.0735), Older construction age.

---

## 🚀 Quickstart & Usage

### 1. Generate Real Estate Data
```bash
python dataset_generator.py
```

### 2. Run Preprocessing Pipeline
```bash
python preprocessor.py
```

### 3. Train & Compare Models
```bash
python train_models.py
```

### 4. Evaluate Test Set & Run Property Valuation
```bash
python evaluate.py
```

---

## 🔮 Future Enhancements
- [ ] Incorporate XGBoost and CatBoost gradient boosting regressor models.
- [ ] Add spatial coordinates with Haversine distance to commercial landmarks.
- [ ] Build a lightweight Streamlit interactive real estate calculator dashboard.
