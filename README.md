# Customer-Segmentation
# 🛍️ Customer Segmentation via Unsupervised Learning

**An end-to-end, unsupervised machine learning pipeline for discovering actionable customer personas from raw retail data — no labels, no ground truth, just math and business logic.**

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![pandas](https://img.shields.io/badge/pandas-2.2.2-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![UMAP](https://img.shields.io/badge/UMAP-0.5.6-8A2BE2?style=flat-square)](https://umap-learn.readthedocs.io/)
[![Jupyter](https://img.shields.io/badge/Jupyter-1.0.0-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## 📑 Table of Contents

- [Business Problem](#-business-problem)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
  - [1. Preprocessing](#1-preprocessing)
  - [2. Dimensionality Reduction](#2-dimensionality-reduction)
  - [3. Clustering](#3-clustering)
  - [4. Visualization](#4-visualization)
- [Results & Personas](#-results--personas)
- [Project Structure](#-project-structure)
- [Installation & Usage](#-installation--usage)
- [Key Engineering Decisions](#-key-engineering-decisions)
- [Future Improvements](#-future-improvements)
- [License](#-license)

---

## 🎯 Business Problem

Imagine you're the Lead Data Scientist for a mid-sized retail company. The marketing team has one goal: launch **3–4 highly targeted campaigns** instead of a single generic, one-size-fits-all promotion. The problem is, nobody actually knows how many distinct types of customers exist in the database, or what separates them.

There's no historical "segment" label to train against — this is a classic **unlabeled, exploratory** problem. The objective of this project is to:

1. Discover natural groupings in customer purchasing and demographic behavior using **unsupervised learning**.
2. Statistically validate *how many* segments genuinely exist in the data (rather than guessing a number).
3. Translate the resulting mathematical clusters into **human-readable, marketable personas** that a non-technical stakeholder can act on immediately.

---

## 🗂️ Dataset

The analysis uses a retail customer records dataset containing **~2,200 rows and 29 raw features**, covering demographics, campaign history, and spending behavior across product categories.

| Category | Example Features |
|---|---|
| **Demographics** | `Year_Birth`, `Education`, `Marital_Status`, `Income`, `Kidhome`, `Teenhome` |
| **Spending (last 2 years)** | `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` |
| **Purchase Behavior** | `NumDealsPurchases`, `NumWebPurchases`, `NumCatalogPurchases`, `NumStorePurchases`, `NumWebVisitsMonth` |
| **Engagement** | `AcceptedCmp1`–`AcceptedCmp5`, `Response`, `Complain`, `Recency` |

Only **numeric, non-constant** features are retained for modeling — categorical fields (e.g., `Education`, `Marital_Status`) are excluded from this baseline pipeline, since PCA and K-Means require a well-defined Euclidean distance, which text categories don't provide.

> ⚠️ Rows with missing values are dropped rather than imputed, keeping this exploratory stage simple and fully auditable. Imputation is deliberately left as a separate downstream modeling decision.

---

## 🔬 Methodology

The pipeline is fully programmatic end-to-end — there are **no hardcoded cluster counts or manually eyeballed thresholds**. Every key parameter (number of PCA components, optimal K, persona labels) is derived directly from the data.

### 1. Preprocessing

- Drop identifier columns (`ID`) — they carry zero behavioral signal and would corrupt distance-based algorithms if left in.
- Drop rows with missing values.
- Filter to numeric-only columns and remove any zero-variance (constant) columns.
- Standardize all remaining features with `StandardScaler` so that high-magnitude features (e.g., `Income`, in the tens of thousands) don't dominate low-magnitude ones (e.g., `NumDealsPurchases`, in single digits) purely due to scale.

### 2. Dimensionality Reduction

With ~25 numeric features, direct clustering suffers from the **Curse of Dimensionality** — pairwise distances become statistically meaningless as dimensionality grows. `PCA` is used to compress the feature space:

- Fit an unconstrained `PCA()` to inspect the full explained-variance spectrum.
- Programmatically select the minimum number of components needed to retain **≥ 80% of cumulative variance** (visualized via a Scree Plot).
- Project the standardized data onto this reduced component space for all downstream modeling.

### 3. Clustering

- Sweep `KMeans` across `n_clusters = 2` through `8`.
- Score each candidate K with the **Silhouette Score** — a more reliable model-selection metric than the Elbow Method, since it doesn't trivially improve as K grows.
- Select the K that **maximizes** the silhouette curve and retrain the definitive model at that value.
- As an advanced extension, a **Gaussian Mixture Model (GMM)** is fit on the same PCA space to obtain *soft*, probabilistic cluster membership — surfacing customers who sit right on the boundary between two personas (a hard, one-hot K-Means assignment can't express this ambiguity).

### 4. Visualization

- A **Scree Plot** validates the PCA component choice.
- A **Silhouette Score curve** validates the K-Means cluster count.
- A **grouped bar chart** compares each cluster's average profile across the key persona-defining features.
- A 2D **UMAP** projection is used for the final presentation-quality visualization — UMAP preserves local neighborhood structure better than PCA, so genuinely distinct clusters appear as visually separated "islands," which is ideal for stakeholder-facing plots.

---

## 📊 Results & Personas

Cluster means are computed on the **original, unscaled data** (not z-scores), so the output is directly interpretable in real business units — dollars and purchase counts, not abstract numbers.

| Cluster | Avg. Income | Avg. Wine Spend | Avg. Meat Spend | Avg. Deals Used | Segment Size | Persona |
|---|---|---|---|---|---|---|
| 0 | High | High | High | Low | ~22% | 🥂 **The Premium Big Spenders** |
| 1 | High | High | High | High | ~19% | 💳 **The Affluent Bargain Hunters** |
| 2 | High | Low | Low | Low | ~24% | 🧐 **The Cautious High-Earners** |
| 3 | Low | Low | Low | High | ~18% | 🏷️ **The Budget-Conscious Deal Hunters** |
| 4 | Low | Low | Low | Low | ~17% | 😴 **The Low-Engagement Segment** |

*(Exact values, cluster count, and segment sizes are generated dynamically by the notebook and will vary slightly depending on the optimal K discovered by the Silhouette Score.)*

**How marketing can act on this:**

- **Premium Big Spenders** → high-margin, low-discount premium offers; loyalty perks over coupons.
- **Affluent Bargain Hunters** → bundle deals and limited-time premium discounts — they respond to value, not just price.
- **Cautious High-Earners** → upsell and awareness campaigns; they have the budget but aren't currently engaged.
- **Budget-Conscious Deal Hunters** → discount-led acquisition and retention offers.
- **Low-Engagement Segment** → re-activation campaigns or exclusion from paid marketing spend to protect ROI.

---

## 📁 Project Structure

```
customer-segmentation-marketing/
├── README.md                       # Project documentation (this file)
├── customer_segmentation.ipynb     # Main analysis notebook (end-to-end pipeline)
└── requirements.txt                # Python dependencies
```

---

## ⚙️ Installation & Usage

### Prerequisites

- Python 3.11+
- `pip` or `conda`

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/customer-segmentation-marketing.git
cd customer-segmentation-marketing
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate      # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.5.0
matplotlib==3.9.0
seaborn==0.13.2
umap-learn==0.5.6
jupyter==1.0.0
```

### 4. Launch the Notebook

```bash
jupyter notebook customer_segmentation.ipynb
```

### 5. Run the Pipeline

Execute the cells in order (`Cell → Run All`). The notebook is organized into four self-contained stages:

```text
1. Data Loading & Preprocessing   →  df_scaled
2. PCA + K-Means Clustering       →  final_kmeans, pca_data
3. Business Interpretation        →  cluster_summary (personas table)
4. Advanced Sandbox (GMM + UMAP)  →  boundary_customers, 2D visualization
```

Each stage prints its own diagnostics (shapes, retained components, optimal K, silhouette scores) so you can verify the pipeline at every step without needing to inspect intermediate variables manually.

---

## 🧠 Key Engineering Decisions

- **Silhouette Score over the Elbow Method** — inertia decreases monotonically with K, forcing a subjective "eyeball the elbow" judgment call. Silhouette Score produces a genuine peak, making the optimal K a deterministic, defensible output rather than a guess.
- **Fully programmatic thresholds** — both the PCA component count and the optimal K are *computed*, not hardcoded. This means the notebook stays correct and reproducible even if it's re-run against a refreshed dataset or a different `random_state`.
- **Personas derived by relative rank, not fixed rules** — persona labels are assigned based on each cluster's position *relative to the other clusters* (via medians), not fixed absolute thresholds. This keeps the labeling logic valid regardless of how cluster numbering shakes out between runs.
- **Zero-variance column guard** — constant columns are detected and dropped before scaling. They carry no clustering signal and would otherwise pass through silently as noisy, uninformative dimensions.
- **Grouping on unscaled data for interpretation** — cluster labels are attached back to the *original* (unscaled) dataframe for the persona summary table. A marketing stakeholder can't act on "Income = 1.4 standard deviations"; they can act on "$78,400 average household income."
- **GMM as a complement to K-Means, not a replacement** — K-Means gives a clean, hard segmentation for campaign targeting, while GMM's `predict_proba()` output identifies ambiguous, "on-the-fence" customers who may need a blended messaging strategy.
- **UMAP reserved for visualization only, never for clustering** — UMAP distorts global distances by design in order to preserve local structure, which makes it excellent for human-readable 2D plots but statistically unreliable as an input space for the clustering algorithm itself. Clustering is always performed on the PCA space.

---

## 🚀 Future Improvements

- [ ] Encode categorical features (`Education`, `Marital_Status`) into the clustering space via one-hot or target encoding.
- [ ] Add cluster stability testing (e.g., bootstrapped re-runs, Adjusted Rand Index across seeds).
- [ ] Benchmark K-Means against `HDBSCAN` for density-based, automatically-determined cluster counts.
- [ ] Expose the persona table and visualizations via a lightweight dashboard (Streamlit or Dash) for the marketing team.
- [ ] Add a scoring function to assign *new, unseen* customers to an existing persona in near real time.
- [ ] Track persona drift over time as customer behavior evolves quarter over quarter.
- [ ] Package the pipeline as an installable module with a CLI entry point (`segment-customers --input data.csv`).

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

If this project was useful to you, consider leaving a ⭐ on the repo!

</div>





