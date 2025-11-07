# Income Classification using Logistic Regression, SVM, and K-Means Clustering

## Overview

This project explores income classification using both supervised and unsupervised machine learning techniques.  
The goal is to predict whether an individual's annual income is greater or less than \$50,000, based on demographic and employment attributes.

Three models are developed and evaluated:

1. **Logistic Regression (LR)**
2. **Support Vector Machine (SVM)**
3. **K-Means Clustering**

Each model is trained, tested, and compared using normalized data, cross-validation, and fine-tuning.

## Dataset

- File: `income.csv`
- Records: 26,215 (after cleaning → 21,537)
- Attributes: 9 original features including age, education, occupation, hours-per-week, and marital status
- Target Variable: `income` (binary: >50K or ≤50K)

### Preprocessing Steps

1. Handle Missing Values: removed rows with missing data
2. Remove Duplicates: ensured each record is unique
3. Encode Categorical Features:
   - Ordinal encoding for _education_
   - Binary encoding for _sex_ (0 = Male, 1 = Female)
   - One-hot encoding for all other categorical features
4. Normalization: scaled all features using MinMaxScaler
5. Data Split:
   - Training: 90%
   - Testing: 10%

## Models and Evaluation

### **1️⃣ Logistic Regression**

- Trained using default and fine-tuned parameters (`penalty`, `C`, `solver`)
- Evaluated with 10-fold cross-validation and test accuracy

| Metric              | Before Tuning | After Tuning |
| ------------------- | ------------- | ------------ |
| Average CV Accuracy | 0.8069        | 0.8069       |
| Test Accuracy       | 0.7776        | 0.7776       |

➡️ The fine-tuned parameters were nearly identical to the default optimal setup.

### **2️⃣ Support Vector Machine (SVM)**

- Explored different kernels (`linear`, `poly`) and hyperparameters (`C`, `degree`, `gamma`)
- Evaluated with 10-fold cross-validation and test accuracy

| Metric              | Before Tuning | After Tuning |
| ------------------- | ------------- | ------------ |
| Average CV Accuracy | 0.8008        | 0.8073       |
| Test Accuracy       | 0.7924        | 0.7966       |

➡️ SVM slightly outperformed Logistic Regression after fine-tuning, achieving the highest accuracy.

### **3️⃣ K-Means Clustering (Unsupervised)**

- Applied on normalized training features (`X_train_norm`)
- Number of clusters: 2
- Compared cluster assignments with true income labels

| Metric         | Value  |
| -------------- | ------ |
| Test Accuracy  | 0.7135 |
| Cluster 0 Size | 10,113 |
| Cluster 1 Size | 9,270  |

### Prototype Comparison:

- Cluster 0 (lower income): younger, female, lower education, fewer working hours, mostly “NotMarried”
- Cluster 1 (higher income): older, male, higher education, longer working hours, mostly “Married”

➡️ K-Means captured general trends but performed worse than the supervised models, as expected.

## Technologies Used

- Python 3
- Libraries:
  - `pandas`, `numpy`, `matplotlib`, `seaborn`
  - `scikit-learn` for model training, normalization, cross-validation, and clustering

## How to Run

```bash
# Clone the repository
git clone https://github.com/your-username/Income-Classification-LR-SVM.git
cd Income-Classification-LR-SVM

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Income_classification.ipynb
```
