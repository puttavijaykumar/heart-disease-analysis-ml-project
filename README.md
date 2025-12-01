# Heart Disease Classification — Machine Learning Project

### End-to-End Modular ML Pipeline | SMOTE-NC | XGBoost | Evaluation Suite

---

## Project Overview

This project builds a complete **machine learning system** to predict the severity of **heart disease** (classes 0–4) using structured medical data. It includes:

- Fully modular **src/** architecture
- Clean preprocessing
- SMOTE-NC oversampling
- Multiple ML models trained & evaluated
- Detailed metric comparison
- Individual confusion matrices
- Professional plots (without seaborn)

The project is designed to be **production-ready**, **reusable**, and **easy to extend**.

---

## Key Features

### Modular Architecture (`src/`)

- `preprocessing.py` → cleaning, encoding, imputing, scaling, SMOTE-NC
- `model_training.py` → trains all ML models
- `evaluation.py` → accuracy, macro/weighted metrics, per-class metrics
- `utils.py` → common helper functions

### Data Preprocessing

- Handled missing values
- Encoded categorical features
- Scaled numeric features
- Applied **SMOTE-NC** to balance all 5 classes

### Models Trained

- Logistic Regression
- Random Forest
- SVM
- KNN
- XGBoost
- CatBoost

### Evaluation Metrics

- Accuracy
- Precision (Macro & Weighted)
- Recall (Macro & Weighted)
- F1-score (Macro & Weighted)
- Per-class P/R/F1
- Confusion matrices
- Comparison bar charts

### Best Model: **XGBoost (78.8%)**

---

## Project Structure

```
heart-disease-classification/
│
├── data/
│   └── heart.csv
│
├── notebook/
│   └── heart_disease_classification.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── utils.py
│
├── models/
│   └── (trained model files)
│
├── reports/
│   ├── confusion_matrices/
│   └── comparison_plots/
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/heart-disease-classification.git
cd heart-disease-classification
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate     # Mac/Linux
venv\Scripts\activate        # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Dataset

The dataset used is located in:

```
data/heart.csv
```

Contains features like:

- age
- sex
- chest pain (cp)
- resting blood pressure
- serum cholesterol
- fasting blood sugar
- max heart rate (thalch)
- slope, ca, thal
- target `num` (0–4)

### Dataset Statistics

- **Samples**: 303 records
- **Features**: 13 numerical & categorical
- **Target Classes**: 5 (0, 1, 2, 3, 4)
- **Missing Values**: Handled with imputation

---

## Data Preprocessing Pipeline

The following steps are implemented:

- Dropping unnecessary columns
- Handling missing values
- Encoding categorical variables
- Scaling numeric features
- Balancing target classes using **SMOTE-NC**

Example call:

```python
from src.preprocessing import load_data, clean_and_encode, apply_smote, scale_split

df = load_data("data/heart.csv")
df_clean = clean_and_encode(df)
X_resampled, y_resampled = apply_smote(df_clean)
X_train, X_test, y_train, y_test = scale_split(X_resampled, y_resampled)
```

---

## Model Training

All models are trained via a single modular call:

```python
from src.model_training import train_models

models = train_models(X_train, y_train)
```

Models include:

- Logistic Regression
- Random Forest
- SVM
- KNN
- XGBoost
- CatBoost

---

## Evaluation & Visualizations

Evaluation returns comprehensive metrics:

```python
from src.evaluation import evaluate_models

results = evaluate_models(models, X_test, y_test)
```

This includes:

- Accuracy
- Macro Precision / Recall / F1
- Weighted Precision / Recall / F1
- Per-class metrics

### Confusion Matrices

Each model gets its own confusion matrix visualization saved in `reports/confusion_matrices/`.

### Comparison Plots

- Accuracy comparison
- Macro/Weighted metric comparison
- Per-class heatmap
- Sorted accuracy chart

All visuals generated using **pure matplotlib** (no seaborn).

---

## Best Model Results

| Model | Accuracy |
|-------|----------|
| **XGBoost** | **0.788** |
| Random Forest | 0.764 |
| CatBoost | 0.749 |
| KNN | 0.654 |
| SVM | 0.635 |
| Logistic Regression | 0.528 |

**XGBoost achieved the overall best performance with 78.8% accuracy.**

---

## Model Performance Breakdown

### XGBoost (Best Model)

- **Accuracy**: 78.8%
- **Macro Precision**: 0.782
- **Macro Recall**: 0.788
- **Macro F1-Score**: 0.784
- **Weighted Precision**: 0.789
- **Weighted Recall**: 0.788
- **Weighted F1-Score**: 0.787

---

## How to Use

### 1. Run the Complete Pipeline

```bash
python main.py
```

### 2. Make Predictions with Best Model

```python
from src.model_training import load_model

model = load_model("models/xgboost_model.pkl")
predictions = model.predict(X_test)
```

### 3. View Reports

All evaluation reports and visualizations are saved in the `reports/` directory.

---

## Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.8+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn, XGBoost, CatBoost |
| **Class Balancing** | Imbalanced-Learn (SMOTE-NC) |
| **Visualization** | Matplotlib |
| **Jupyter** | Jupyter Notebook |

---

## Learning Outcomes

Through this project, I gained hands-on experience in:

- Building modular, production-ready ML code
- Handling imbalanced multiclass classification problems
- Implementing SMOTE-NC for effective class balancing
- Training and comparing multiple ML algorithms
- Creating comprehensive evaluation metrics
- Generating professional visualizations
- Following best practices in code organization

---

## How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. **Clone and navigate to the repository**

```bash
git clone https://github.com/yourusername/heart-disease-classification.git
cd heart-disease-classification
```

2. **Create and activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Jupyter notebook or Python scripts**

```bash
jupyter notebook notebook/heart_disease_classification.ipynb
```

Or run the pipeline:

```bash
python src/main.py
```

---

## Future Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- Add SHAP explainability for model interpretability
- Deploy the model using:
  - FastAPI
  - Streamlit / Gradio
- Build a Docker container for easy deployment
- Add automated tests (pytest)
- Implement cross-validation for robust evaluation
- Add feature importance analysis

---

## Contributing

Contributions are welcome! Please feel free to:

- Fork the repository
- Create a feature branch
- Submit a pull request

---

## License

This project is open source and available under the MIT License. See the LICENSE file for details.

---

## Author

**Putta Vijay Kumar**

AI/ML Engineer | Research Intern  
Central University of Rajasthan

---

## Acknowledgements

- Thanks to the Kaggle community for the heart disease dataset
- Special thanks to the open-source ML community for amazing libraries like Scikit-Learn, XGBoost, and CatBoost

---

**Happy Learning! **