import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import pickle
import os

print("1. Generating clinical dataset with biological interactions...")
np.random.seed(42)
n = 3000

df = pd.DataFrame({
    'Age': np.random.randint(20, 80, n),
    'BMI': np.random.normal(27.5, 5.5, n).clip(16.0, 45.0).round(1),
    'BloodPressure': np.random.normal(125, 18, n).clip(85, 185).round(0),
    'Glucose': np.random.normal(115, 32, n).clip(65, 240).round(0),
    'SleepHours': np.random.normal(6.8, 1.3, n).clip(3.5, 10.0).round(1),
    'PhysicalActivity': np.random.binomial(1, 0.45, n)
})

# Non-linear risk probabilities (logistic sigmoid transformation with noise)
def generate_target(score_series):
    prob = 1 / (1 + np.exp(-score_series))
    return np.random.binomial(1, prob)

# Continuous risk equations centered around clinical thresholds
d_score = (df['Glucose'] - 130) * 0.08 + (df['BMI'] - 27) * 0.22 - df['PhysicalActivity'] * 0.8 + np.random.normal(0, 0.5, n)
h_score = (df['BloodPressure'] - 132) * 0.09 + (df['Age'] - 50) * 0.06 + (df['BMI'] - 28) * 0.15 - df['PhysicalActivity'] * 0.9 + np.random.normal(0, 0.5, n)
k_score = (df['BloodPressure'] - 135) * 0.08 + (df['Glucose'] - 135) * 0.06 + (6.0 - df['SleepHours']) * 0.55 + np.random.normal(0, 0.5, n)

df['DiabetesRisk'] = generate_target(d_score)
df['HeartRisk'] = generate_target(h_score)
df['KidneyRisk'] = generate_target(k_score)

os.makedirs('data', exist_ok=True)
df.to_csv('data/health_data.csv', index=False)

features = ['Age', 'BMI', 'BloodPressure', 'Glucose', 'SleepHours', 'PhysicalActivity']
X = df[features]

targets = {
    'diabetes': 'DiabetesRisk',
    'heart': 'HeartRisk',
    'kidney': 'KidneyRisk'
}

best_models = {}
model_metadata = {}
report_lines = ["=== MEDISENSE AI: ADVANCED MULTI-MODEL BENCHMARK REPORT ===\n"]

print("\n2. Running Model Tournament (Logistic Regression vs Random Forest vs Gradient Boosting)...")
for key, target_col in targets.items():
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    candidates = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42)
    }
    
    best_auc = 0
    best_name = ""
    best_clf = None
    
    print(f"\n--- Evaluating Target: {key.upper()} ---")
    report_lines.append(f"\n--- Target: {key.upper()} ---")
    
    for name, clf in candidates.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred) * 100
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cv_auc = cross_val_score(clf, X, y, cv=5, scoring='roc_auc').mean()
        
        line = f"  {name:20s} | Acc: {acc:5.2f}% | F1: {f1:.3f} | Test AUC: {auc:.3f} | 5-Fold CV AUC: {cv_auc:.3f}"
        print(line)
        report_lines.append(line)
        
        if auc > best_auc:
            best_auc = auc
            best_name = name
            best_clf = clf
            
    print(f"  --> Winner for {key.upper()}: {best_name} (ROC-AUC: {best_auc:.3f})")
    report_lines.append(f"  --> Selected Production Model: {best_name} (ROC-AUC: {best_auc:.3f})")
    best_models[key] = best_clf
    model_metadata[key] = {'algorithm': best_name, 'roc_auc': round(best_auc, 3)}

print("\n3. Training Unsupervised K-Means Clustering for Patient Phenotyping...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# Save models, scaler, clustering, and metadata together
best_models['scaler'] = scaler
best_models['kmeans'] = kmeans
best_models['metadata'] = model_metadata

os.makedirs('models', exist_ok=True)
with open('models/multi_health_models.pkl', 'wb') as f:
    pickle.dump(best_models, f)

with open('models/evaluation_metrics.txt', 'w') as f:
    f.write("\n".join(report_lines))

print("\nSuccess! Best models, K-Means clusterer, and benchmark report saved to 'models/'.")