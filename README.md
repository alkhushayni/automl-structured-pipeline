# AutoML for Structured Data: A Configurable Pipeline with Transparent Feature Engineering and Real-World Validation

This repository contains the implementation of a modular AutoML framework developed for the automation of machine learning pipelines on structured tabular data. The system integrates key preprocessing steps, feature engineering using Variance Inflation Factor (VIF), classifier benchmarking, and output formatting for real-world integration.

> 🔍 This code accompanies the manuscript:  
> **"AutoML for Structured Data: A Configurable Pipeline with Transparent Feature Engineering and Real-World Validation"**  
> Submitted to *Array* (Elsevier), 2025.

---

## 📁 Repository Structure

```
.
├── notebooks/              # Jupyter notebooks for data flow and evaluation
│   ├── classification.ipynb
│   ├── data collection.ipynb
│   ├── data preparation.ipynb
│   └── vif.ipynb
├── datasets/               # CSV-formatted datasets used in the study
│   ├── heart.csv
│   ├── bank.csv
│   ├── adult.data
│   └── adult.test
├── src/                    # Python scripts and settings
│   ├── settings.py
│   ├── __init__.py
│   └── tests/
│       └── test_data.py
├── requirements.txt        # Required packages (scikit-learn, pandas, etc.)
├── LICENSE
└── README.md
```

---

## 📊 Datasets Used

All datasets are public and preprocessed versions are included in the `/datasets` folder.

- [Heart Disease Dataset (UCI)](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- [Adult Income Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/adult)
- [Bank Marketing Dataset (UCI)](https://archive.ics.uci.edu/ml/datasets/bank+marketing)

---

## 🧪 How to Use

1. 📦 Install dependencies:

```bash
pip install -r requirements.txt
```

2. 📁 Place any additional datasets into the `/datasets` directory.

3. 📓 Open the notebooks in `/notebooks`:

- `data collection.ipynb` → Load and explore datasets
- `data preparation.ipynb` → Clean and format data
- `vif.ipynb` → Apply VIF-based feature filtering
- `classification.ipynb` → Train models and evaluate performance

4. 🧠 Modify dataset name and target column inside notebooks as needed:

```python
dataset_name = "heart.csv"
target_column = "target"
```

---

## 🧾 Citation

If you use this code in your work, please cite our paper:

> Alkhushayni, S. et al. (2025). *AutoML for Structured Data: A Configurable Pipeline with Transparent Feature Engineering and Real-World Validation*. Under review, Array (Elsevier).

---

## 👥 Authors

- Suboh Alkhushayni (Yarmouk University, Jordan)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
