# Amazon ML Challenge 2025
## 1. Executive Summary

We approached the pricing challenge by using **textual and numeric features** extracted from the product `catalog_content`. Our key innovation involved **structured feature engineering from `catalog_content`**, creating numeric representations like `unit_size`, `pack_count`, and `total_size` to complement the textual TF-IDF embeddings. These features are standardized to ounces and log-transformed for modeling stability.
<br>
The model achieves a validation SMAPE of 50.32%.
The final predictive model is a **LightGBM regression** trained on this combined feature set.

---

## 2. Methodology Overview

### 2.1 Problem Analysis

- The objective was to predict product prices based on catalog listings that include textual descriptions and image links.
- During exploratory analysis, we observed highly skewed price distribution, prompting the use of log transformation (log1p) for modeling stability.
- Also many items contained implicit information about **brand, unit size, and pack count** embedded in the text. Extracting these structured features provided additional signals for price prediction beyond raw text.

**Key Observations:**

- Catalog text contains explicit and implicit product measurements (e.g., “12 oz”, “pack of 6”).
- Brand names often appear as the first word of the item name.
- Many numeric quantities required unit standardization (e.g., grams → ounces).

### 2.2 Solution Strategy

- We adopted a single-model regression approach that combines structured numeric features with TF-IDF text embeddings.
- Custom regex-based parsing and unit normalization were used to extract **brand_name, unit_size, pack_count, and total_size** from `catalog_content`.
- Numeric features were standardized to ounces, log-transformed, and scaled using **StandardScaler**.
- TF-IDF vectorization was applied to the catalog text using **50,000 features and ngram range (1,2)**.
- Combined features were fed into a **LightGBM regression model** for log-price prediction.
- The pipeline is modular and designed for future integration of image embeddings.

**Approach Type:** Single Model (Text + Numeric Features)
**Core Innovation:** A custom feature extraction engine that parses noisy catalog text to derive standardized physical quantity features such as **unit size, pack count, and total size** which are then combined with TF-IDF embeddings to enable accurate and scalable price prediction.

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
Catalog Content + Numeric Columns
           │
           ▼
   ┌─────────────────┐
   │ Text Preprocessing│
   │  - TF-IDF (50k) │
   │  - ngram (1,2)  │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │ Numeric Features │
   │  - unit_size     │
   │  - pack_count    │
   │  - total_size    │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │ Feature Concatenation │
   └─────────────────┘
           │
           ▼
   ┌─────────────────┐
   │  LightGBM Model │
   │  Regression     │
   └─────────────────┘
           │
           ▼
     Predicted Price
```

### 3.2 Model Components

**Text Processing Pipeline:**

- Preprocessing steps: Fill missing text, TF-IDF vectorization with `max_features=50,000` and `ngram_range=(1,2)`.
- Model type: LightGBM regression on sparse text matrix.
- Key parameters: `learning_rate=0.05`, `num_leaves=63`, `feature_fraction=0.8`, `bagging_fraction=0.8`, `num_boost_round=2000`, early stopping with 100 rounds.

**Image Processing Pipeline:**

- Yet to do

---

## 4. Model Performance

### 4.1 Validation Results

- **SMAPE Score:** 50.32% (best validation)

---

## 5. Conclusion

By combining **structured feature extraction** from product catalogs with **LightGBM regression** on TF-IDF and numeric features, we achieved a robust price prediction pipeline. This approach effectively leveraged hidden numeric information embedded in text, providing significant improvement over simple baseline models using only catalog_content.
<br>
Image embeddings were planned but not yet integrated. Future work can extend the model to multimodal inputs for further gains.
