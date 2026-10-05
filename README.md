# 🏆 Datathon All-in-One Template

Tabular · Image (Computer Vision) · Text (NLP) — তিন ধরনের data-র জন্য সম্পূর্ণ, চালানোর মতো (run-ready) Jupyter notebook template।
প্রতিটা notebook-এ **data না দিলে নিজেই একটা messy demo dataset বানিয়ে নেয়**, তাই এখনই practice করা যাবে।

```
Dathathon/
├── 01_tabular_template.ipynb   ← CSV/Excel data: classification + regression
├── 02_image_template.ipynb     ← Image classification (PyTorch + timm)
├── 03_nlp_template.ipynb       ← Text classification (TF-IDF → Transformers, English + বাংলা)
├── 04_timeseries_template.ipynb ← Forecasting (sales/demand): baselines, ETS/SARIMA, LightGBM lag model
├── 05_audio_template.ipynb      ← Audio classification: MFCC features, AST embedding, mel-spectrogram CNN
├── requirements.txt
└── README.md                   ← এই guide + cheatsheet
```

---

## 🚀 কিভাবে ব্যবহার করবো

### ☁️ Google Colab (recommended)
1. **Notebook খোলো:** [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → `.ipynb` file select করো
   (অথবা notebook গুলো Google Drive এ রেখে double-click → *Open with Google Colaboratory*)
2. **GPU চালু করো** (image / transformer এর জন্য): **Runtime → Change runtime type → T4 GPU → Save**
3. প্রথম cell **"☁️ Google Colab Setup"** এ `DATA_SOURCE` বদলাও:

   | `DATA_SOURCE` | কখন | কি করতে হবে |
   |---|---|---|
   | `"demo"` | Practice | কিছু না — শুধু Run all |
   | `"upload"` | ছোট CSV file | Cell চালালে **Choose Files** → train/test/sample_submission (বা `.zip`) |
   | `"drive"` | বড় data / image folder | Drive এ `MyDrive/datathon/tabular/` (বা `image/`, `nlp/`, `timeseries/`) এ data রাখো → permission দাও |
   | `"kaggle"` | Kaggle competition | kaggle.com → Settings → **Create New Token** → `kaggle.json` upload; `KAGGLE_COMPETITION = "titanic"` |

   Data automatically `data/tabular` / `data/image` / `data/nlp` / `data/timeseries` এ চলে আসে, `.zip` থাকলে নিজেই unzip হয়।
4. **CONFIG cell** এ file আর column এর নাম মিলাও (`TARGET`, `ID_COL`, `METRIC`, image হলে `IMG_COL`, `LABEL_COL`, text হলে `TEXT_COL`)
5. **Runtime → Run all** (`Ctrl+F9`)
6. শেষ cell **submission.csv auto download** করবে। `SAVE_TO_DRIVE = True` দিলে model আর outputs Drive এ backup হবে।

**Colab tips**
- Colab বন্ধ / disconnect হলে `/content` এর সব file **মুছে যায়** → গুরুত্বপূর্ণ জিনিস Drive এ save করো
- ~90 মিনিট কিছু না করলে disconnect হয় — লম্বা training এর সময় tab খোলা রাখো
- Free GPU এর সময় সীমিত — EDA / tabular এর সময় GPU বন্ধ রাখো (Runtime type → CPU), শুধু image/transformer এ চালু করো
- `CUDA out of memory` → `BATCH_SIZE` কমাও, অথবা **Runtime → Restart session**
- Library install এর পর error এলে **Runtime → Restart session** করে আবার চালাও

### Local
```bash
pip install -r requirements.txt
```
```bash
jupyter notebook
```

### Kaggle Notebook
1. Notebook upload করো (*File → Import Notebook*), Colab Setup cell এ `DATA_SOURCE = "demo"` রেখে দাও (missing library নিজেই install হবে)
2. **CONFIG cell** এ শুধু path আর column নাম বদলাও:
   ```python
   DATA_DIR = Path("/kaggle/input/<competition-name>")
   TARGET = "your_target_column"
   ID_COL = "id"
   METRIC = "f1_macro"      # competition যে metric এ judge করবে
   ```
3. Image/NLP transformer এর জন্য **GPU ON** + **Internet ON**
4. *Run All*

> 💡 Practice: কোনো data ছাড়াই *Run All* দাও → demo data দিয়ে পুরো pipeline চলবে। তারপর Kaggle এর
> Titanic / House Prices / Spaceship Titanic / Digit Recognizer / NLP Disaster Tweets /
> Store Sales (time series: `store-sales-time-series-forecasting`) এ CONFIG বদলে চালাও।

---

## 🎯 Practice Datasets (Kaggle) — notebook অনুযায়ী

Colab Setup এ `DATA_SOURCE = "kaggle"` → competition হলে `KAGGLE_COMPETITION`, dataset হলে `KAGGLE_DATASET` দাও।
Competition এর data নামানোর আগে Kaggle page এ **Join / I Understand and Accept** চাপতে হবে।

| Notebook | Dataset (Kaggle নাম) | Task | CONFIG এ কি দেবে |
|---|---|---|---|
| 01 Tabular | `titanic` ⭐ শুরু | binary, missing value অনেক | `TARGET="Survived"`, `ID_COL="PassengerId"`, `METRIC="accuracy"` |
| 01 Tabular | `spaceship-titanic` | binary, feature engineering | `TARGET="Transported"`, `ID_COL="PassengerId"`, `METRIC="accuracy"` |
| 01 Tabular | `house-prices-advanced-regression-techniques` | regression, skewed target | `TARGET="SalePrice"`, `ID_COL="Id"`, `METRIC="neg_root_mean_squared_error"` (log target) |
| 02 Image | `aerial-cactus-identification` | binary image, CSV format | `IMG_COL="id"`, `LABEL_COL="has_cactus"`, train.zip unzip → `TRAIN_IMG_DIR` ঠিক করো |
| 02 Image | `plant-seedlings-classification` | 12 class, folder format | `DATA_FORMAT="folder"` |
| 03 NLP | `nlp-getting-started` ⭐ (Disaster Tweets) | binary text | `TEXT_COL="text"`, `TARGET="target"`, `METRIC="f1"` |
| 03 NLP | `spooky-author-identification` | 3 class text | `TEXT_COL="text"`, `TARGET="author"`, `METRIC="neg_log_loss"` |
| 04 Time series | `demand-forecasting-kernels-only` ⭐ | 50 item × 10 store daily | `ID_COLS=["store","item"]`, `TARGET="sales"`, `METRIC="smape"` |
| 04 Time series | `store-sales-time-series-forecasting` | ১৭৮২ series + promo | `ID_COLS=["store_nbr","family"]`, `METRIC="rmsle"` (বড় data) |
| 05 Audio | `freesound-audio-tagging` | 41 class sound (বড়) | `FILE_COL="fname"`, `TRAIN_AUDIO_DIR=DATA_DIR/"audio_train"`, `TEST_AUDIO_DIR=DATA_DIR/"audio_test"`, `MAX_TRAIN_CLIPS=3000` |

> নাম কখনো বদলে গেলে / download না হলে Kaggle এ search করে URL এর শেষ অংশটা নাম হিসেবে দাও।

---

## ⏱️ Competition দিনের Strategy (time plan)

| সময় | কাজ |
|---|---|
| প্রথম ১৫ মিনিট | Problem statement, **evaluation metric**, submission format ভালো করে পড়ো |
| ১৫–৩০ মিনিট | Data load + First Look + **⚡ Quick Baseline submit** (safety net) |
| ৩০ মিনিট – ২ ঘণ্টা | Cleaning + EDA → insight খোঁজো (missing pattern, leakage, distribution shift) |
| ২ – ৪ ঘণ্টা | **Feature engineering** (সবচেয়ে বেশি score বাড়ে এখানে) + Model zoo |
| ৪ – ৫ ঘণ্টা | Top 2-3 model tune (Optuna) + K-Fold OOF |
| শেষ ১ ঘণ্টা | Ensemble, threshold tune, **submission format check**, presentation/plot |

**Golden rules**
- সবসময় **CV score** বিশ্বাস করো, public leaderboard-এর পেছনে বেশি দৌড়াবে না (overfit to LB)
- প্রতিটা ভালো submission এর পর notebook **version save** করো
- একসাথে একটা জিনিস বদলাও → CV দেখো → রাখো/বাদ দাও

---

## 🔄 Universal Workflow

```
Load → First Look → Clean (missing, duplicate, type, outlier) → EDA → Feature Engineering
     → Encode/Scale → Feature Selection → CV Model Zoo → Tune → K-Fold OOF → Ensemble → Submit
```

---

## 🧠 কোন Model কখন?

### Tabular
| Model | কখন | Scaling লাগে? |
|---|---|---|
| **LightGBM / XGBoost / CatBoost** | প্রায় সব tabular competition এর winner | ❌ |
| CatBoost | অনেক categorical column | ❌ |
| RandomForest / ExtraTrees | ভালো baseline, কম tuning | ❌ |
| Logistic / Linear / Ridge / Lasso | baseline, interpretability, ছোট data | ✅ |
| SVM / KNN | ছোট data (<20k row) | ✅ |
| MLP | বড় data, ensemble এ diversity আনতে | ✅ |

### Image
| Approach | কখন |
|---|---|
| Pretrained CNN embedding + LogReg | খুব কম data / কম সময় / GPU নেই |
| Fine-tune `resnet` / `efficientnet` / `convnext` | standard (GPU) |
| ViT / EVA / Swin | বড় data, শক্তিশালী GPU |

### Time series
দেখো নিচের **📈 Time Series guide** অংশ।

### NLP
| Approach | কখন |
|---|---|
| TF-IDF (word+char) + LogReg / LinearSVC | **সবসময় আগে এটা** — দ্রুত, শক্তিশালী baseline |
| Sentence embeddings + LogReg | কম data, semantic task |
| Fine-tune BERT / DeBERTa / XLM-R / BanglaBERT | GPU থাকলে best score |

---

## 📏 Metric Cheatsheet (sklearn `scoring` নাম)

| Task | Metric | `scoring=` | Note |
|---|---|---|---|
| Binary classification | Accuracy | `accuracy` | balanced data হলে |
| | F1 | `f1` | imbalanced; threshold tune করো |
| | ROC-AUC | `roc_auc` | probability submit করো |
| | Log loss | `neg_log_loss` | probability, calibration জরুরি |
| Multiclass | Macro F1 | `f1_macro` | প্রতিটা class সমান গুরুত্ব |
| | Weighted F1 | `f1_weighted` | class size অনুযায়ী |
| Regression | RMSE | `neg_root_mean_squared_error` | বড় error কে বেশি শাস্তি |
| | MAE | `neg_mean_absolute_error` | outlier robust |
| | R² | `r2` | |
| | RMSLE | — | `log1p(y)` তে train করো, RMSE optimize করো |
| Forecasting | MAPE / sMAPE | — | % error; target এ zero থাকলে sMAPE/WAPE |
| | WAPE | — | মোট error / মোট actual — sales এ খুব common |

---

## 🧹 Missing value — কোনটা কখন?

| পরিস্থিতি | সমাধান |
|---|---|
| > 60% missing | Column drop (তবে `is_missing` flag রেখে দাও) |
| Numeric, outlier আছে | **Median** |
| Numeric, normal distribution | Mean |
| Feature গুলোর মধ্যে relation আছে | `KNNImputer` / `IterativeImputer` |
| Categorical | Mode বা `"missing"` নামে নতুন category |
| Group অনুযায়ী আলাদা (e.g. city-wise income) | `groupby().transform("median")` |
| Tree boosting model | NaN রেখেই দাও — নিজেই handle করে |

---

## ⚠️ Data Leakage — সবচেয়ে বড় ভুল

1. **Scaler / imputer / encoder পুরো data তে fit করা** → সবসময় `Pipeline` এর ভেতরে রাখো (template এ তাই করা আছে)
2. **Target encoding** CV ছাড়া → `sklearn.preprocessing.TargetEncoder` (internal cross-fitting) ব্যবহার করো
3. **একই user/patient এর row train ও validation দুই জায়গায়** → `GroupKFold`
4. **Time series এ random split** → `TimeSeriesSplit`; ভবিষ্যতের data দিয়ে feature বানাবে না
5. **SMOTE validation fold এ** → `imblearn.pipeline.Pipeline` ব্যবহার করো
6. Test এর row কখনো drop করবে না (outlier হলেও)

---

## 📈 Time Series — দ্রুত guide (`04_timeseries_template.ipynb`)

**CONFIG এ যা বদলাবে:** `DATE_COL`, `TARGET`, `ID_COLS` (store/item; একটা series হলে `[]`), `FREQ`, `SEASON`, `METRIC`

| Data | `FREQ` | `SEASON` |
|---|---|---|
| Daily | `"D"` | 7 |
| Weekly | `"W"` | 52 |
| Monthly | `"MS"` | 12 |
| Hourly | `"h"` | 24 |

| Model | কখন |
|---|---|
| Seasonal naive / seasonal average | **সবসময় baseline** — এটাকে না হারালে কিছু ভুল আছে |
| ETS (Holt-Winters) / SARIMA / Prophet | অল্প series, কম data, monthly data |
| LightGBM + lag/rolling/calendar feature (global model) | অনেক series, daily data, promo/price এর মত extra feature — **competition winner** |
| Ensemble (LightGBM + ETS/seasonal avg) | সবচেয়ে stable |

**Time series এর নিয়ম**
- Validation = **train এর শেষ অংশ** (test যত লম্বা, তত লম্বা) — random split কখনো না
- Lag ≥ forecast horizon (`H`) রাখলে leakage হয় না (notebook এ auto)
- Missing date আগে পূরণ করো, না হলে `lag_7` আসলে ৭ দিন আগের হয় না
- Metric: sales → RMSE/RMSLE/WAPE; অনেক zero থাকলে MAPE ব্যবহার করবে না (০ দিয়ে ভাগ)

---

## 🛠️ দরকারি ছোট snippet

```python
# বড় CSV তে memory কমাও
def reduce_mem(df):
    for c in df.select_dtypes("number"):
        df[c] = pd.to_numeric(df[c], downcast="integer" if df[c].dtype.kind == "i" else "float")
    return df

# Regression target skewed হলে
model.fit(X, np.log1p(y));  pred = np.expm1(model.predict(X_test))

# Seed averaging (stable score)
preds = np.mean([LGBMClassifier(random_state=s).fit(X, y).predict_proba(X_test)[:, 1] for s in range(5)], axis=0)

# Rank average ensemble (AUC metric এর জন্য ভালো)
from scipy.stats import rankdata
ens = (rankdata(p1) + rankdata(p2)) / 2 / len(p1)
```

---

## ❓ Troubleshooting

| সমস্যা | সমাধান |
|---|---|
| `ModuleNotFoundError` | `pip install <name>` — optional library না থাকলে notebook সেই অংশ auto skip করে |
| LightGBM "Do not support special JSON characters in feature name" | column নাম clean করো (tabular template এ `clean_name()` আছে) |
| CUDA out of memory | `BATCH_SIZE` কমাও, `IMG_SIZE`/`MAX_LEN` কমাও |
| Windows এ DataLoader আটকে যায় | `NUM_WORKERS = 0` |
| CV ভালো কিন্তু LB খারাপ | Adversarial validation দেখো (tabular Section 5.6), leakage চেক করো |
| Submission reject | `sample_submission` এর column নাম, row order, dtype হুবহু মিলাও |
