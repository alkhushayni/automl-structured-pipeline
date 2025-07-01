
# AutoML System for Structured Data

This repository contains the source code, datasets, and evaluation results for our proposed AutoML pipeline, as described in the manuscript submitted to *Array*.

## 📄 Project Overview

This AutoML system:
- Supports structured tabular data
- Performs feature engineering, model training, and evaluation
- Outputs models in formats suitable for deployment (.pkl, .json, .yml)
- Includes experimental results across 3 real-world datasets using 6 classifiers

## 📦 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## 📁 Datasets

Place all datasets inside the `datasets/` folder:

```bash
datasets/
├── heart.csv
├── adult.data
├── adult.test
└── bank.csv
```

## 🚀 Run Experiments

To evaluate all classifiers (Random Forest, Decision Tree, KNN, Logistic Regression, Naive Bayes, SVM) on the 3 datasets, run:

```bash
python run_multi_dataset_experiments.py
```

This will generate a results file:

```
automl_multi_dataset_results.csv
```

## 📊 Output

The output CSV includes:
- Dataset name
- Classifier name
- Accuracy, Precision, Recall, F1 Score (in %)

## 📂 Repository Structure

```
AutoML-System/
├── run_multi_dataset_experiments.py
├── requirements.txt
├── README.md
├── LICENSE
├── automl_multi_dataset_results.csv
└── datasets/
    ├── heart.csv
    ├── adult.data
    ├── adult.test
    └── bank.csv
```

## ⚖️ License

This project is licensed under the terms of the LICENSE file (MIT/Apache/Other).

## 📬 Contact

For questions, contact the corresponding author listed in the manuscript.

---

> 📢 This project supports reproducibility and has been structured for transparent benchmarking.
