
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Define classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'KNN': KNeighborsClassifier(n_neighbors=10),
    'Logistic Regression': LogisticRegression(max_iter=200),
    'Naive Bayes': GaussianNB(),
    'SVM': SVC(kernel='rbf', probability=True)
}

# Dataset file paths
datasets_info = {
    'Heart': './datasets/heart.csv',
    'Adult': './datasets/adult.data',
    'Bank': './datasets/bank.csv'
}

# Placeholder for results
results = []

for dataset_name, path in datasets_info.items():
    try:
        df = pd.read_csv(path)

        # Custom preprocessing per dataset
        if dataset_name == 'Adult':
            df.columns = [
                'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status',
                'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss',
                'hours_per_week', 'native_country', 'income'
            ]
            df = df.select_dtypes(include=['number'])
            df = df.dropna()
            df = df.sample(n=5000, random_state=1)  # downsample
            df['target'] = (df['income'] == ' >50K').astype(int) if 'income' in df else df.iloc[:, -1]
        elif dataset_name == 'Bank':
            df = df.select_dtypes(include=['number'])
            df = df.dropna()
            df['target'] = df.iloc[:, -1]
        else:
            df = df.dropna()

        # Prepare features and target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

        for model_name, model in classifiers.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            results.append({
                'Dataset': dataset_name,
                'Model': model_name,
                'Accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
                'Precision': round(precision_score(y_test, y_pred, average='weighted') * 100, 2),
                'Recall': round(recall_score(y_test, y_pred, average='weighted') * 100, 2),
                'F1 Score': round(f1_score(y_test, y_pred, average='weighted') * 100, 2)
            })

    except Exception as e:
        print(f"Error processing {dataset_name}: {e}")

# Save results to CSV
results_df = pd.DataFrame(results)
results_df.to_csv("automl_multi_dataset_results.csv", index=False)
print("Results saved to 'automl_multi_dataset_results.csv'")
