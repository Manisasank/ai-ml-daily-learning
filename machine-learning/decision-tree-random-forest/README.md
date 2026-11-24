# Decision Tree & Random Forest Classifier (From Scratch in NumPy)

A mathematical, object-oriented implementation of recursive **Binary Decision Trees** and the **Random Forest Ensemble** with Bagging, random feature subsampling ($\sqrt{D}$), and Out-of-Bag (OOB) validation.

---

## 📌 Project Overview

Decision Trees are non-parametric supervised learning models that recursively partition feature space into orthogonal hyperplanes. While single trees can be prone to high variance and overfitting, **Random Forest** ensembles mitigate this by combining bootstrap aggregation (Bagging) with random feature subspace selection.

### Key Capabilities:
- **Impurity Criteria:** Analytical implementations of both **Gini Impurity** and **Shannon Entropy** (Information Gain).
- **Pre-Pruning Guardrails:** Configurable `max_depth`, `min_samples_split`, and `min_impurity_decrease` to curb overfitting.
- **Random Forest Ensemble:** Independent bootstrap sampling ($N$ samples with replacement) and split-level feature decorrelation ($k = \sqrt{D}$).
- **Out-of-Bag (OOB) Scoring:** Built-in internal cross-validation evaluating trees only on samples withheld from their bootstrap bags ($~36.8\%$ of data).
- **Mean Decrease in Impurity (MDI):** Feature importance tracking accumulated across splits.
- **ASCII Tree Visualizer:** Direct console inspection of tree branching rules and sample class distributions.

---

## 🏗️ Architecture

```
machine-learning/decision-tree-random-forest/
├── criterion.py       # Gini impurity, Shannon entropy, and candidate split search
├── decision_tree.py   # Recursive binary decision tree classifier with pruning
├── random_forest.py   # Bagging ensemble, feature subsampling, and OOB tracking
├── evaluate_trees.py  # Model benchmarking, MDI feature ranking, & ASCII visualizer
└── README.md          # Theoretical derivations, formulas, and usage guide
```

---

## 📐 Mathematical Foundations

### 1. Impurity Metrics
For a node containing samples distributed across classes $i \in \{1, \dots, C\}$ with empirical probabilities $p_i$:

- **Gini Impurity:**
  $$I_G(p) = 1 - \sum_{i=1}^{C} p_i^2$$

- **Shannon Entropy:**
  $$H(p) = -\sum_{i=1}^{C} p_i \log_2(p_i)$$

### 2. Information Gain (Impurity Reduction)
Given parent dataset $D$ partitioned into left child $D_L$ and right child $D_R$ by threshold $t$:
$$\Delta I(D, t) = I(D) - \left( \frac{|D_L|}{|D|} I(D_L) + \frac{|D_R|}{|D|} I(D_R) \right)$$

### 3. Random Forest Variance Reduction
For $B$ de-correlated trees, each with variance $\sigma^2$ and pairwise correlation $\rho$:
$$\operatorname{Var}\left(\frac{1}{B} \sum_{b=1}^{B} T_b(x)\right) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$
*As $B \to \infty$, the second term approaches zero, and random feature subsampling drives down the correlation $\rho$, dramatically reducing overall ensemble variance.*

### 4. Mean Decrease in Impurity (MDI)
The importance of feature $f$ is the sum of impurity reductions across all internal nodes $t$ where feature $f$ was chosen:
$$\text{Importance}(f) = \frac{1}{B} \sum_{b=1}^{B} \sum_{t \in T_b : v(t) = f} \frac{N_t}{N} \Delta I(t)$$

---

## 📊 Benchmark Results (Wine Classification Dataset)

Evaluated on held-out test partition ($N=45$ samples):

| Model | Test Accuracy | Weighted F1-Score | OOB Validation Score |
| :--- | :---: | :---: | :---: |
| **Single Decision Tree** (`max_depth=5`) | 91.11% | 0.9111 | N/A |
| **Random Forest** ($B=40$ trees, $\sqrt{D}$) | **100.00%** | **1.0000** | **98.50%** |

### Top 5 Predictive Features (MDI Score):
1. **`proline`**: 0.1550
2. **`color_intensity`**: 0.1530
3. **`alcohol`**: 0.1525
4. **`flavanoids`**: 0.1169
5. **`od280/od315_of_diluted_wines`**: 0.1100

---

## 🚀 Quickstart & Usage

### 1. Test Impurity Functions & Split Search
```bash
python criterion.py
```

### 2. Test Recursive Decision Tree Classifier
```bash
python decision_tree.py
```

### 3. Test Random Forest Ensemble with OOB Evaluation
```bash
python random_forest.py
```

### 4. Run Benchmark Comparison & ASCII Tree Visualizer
```bash
python evaluate_trees.py
```

---

## 🔮 Future Enhancements
- [ ] Implement Gradient Boosted Decision Trees (GBDT).
- [ ] Add Cost-Complexity Pruning (CCP) with cross-validated $\alpha$.
- [ ] Extend split criteria to Regression Trees (Mean Squared Error & Variance Reduction).
