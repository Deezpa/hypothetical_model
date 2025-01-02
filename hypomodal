import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix

# Hypothetical Data Generation (as dataset reference [20] is not accessible directly)
def generate_hypothetical_data(n_samples=500):
    np.random.seed(42)
    data = {
        'credit_utilization': np.random.uniform(0, 1, n_samples),
        'payment_history': np.random.choice([0, 1], size=n_samples, p=[0.3, 0.7]),
        'avg_monthly_spend': np.random.normal(500, 150, n_samples),
        'mobile_data_usage': np.random.normal(2, 0.5, n_samples),
        'risk_tolerance': np.random.uniform(0, 1, n_samples),
        'financial_literacy': np.random.uniform(0, 1, n_samples),
        'target': np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])  # Binary creditworthiness target
    }
    return pd.DataFrame(data)

# Load Hypothetical Data
data = generate_hypothetical_data()

# Separate features and target
X = data.drop('target', axis=1)
y = data['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
rf_model = RandomForestClassifier(random_state=42)
gb_model = GradientBoostingClassifier(random_state=42)

# Train and Evaluate Random Forest
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Train and Evaluate Gradient Boosting
gb_model.fit(X_train_scaled, y_train)
y_pred_gb = gb_model.predict(X_test_scaled)
y_prob_gb = gb_model.predict_proba(X_test_scaled)[:, 1]

# Metrics Calculation
def evaluate_model(y_true, y_pred, y_prob, model_name):
    print(f"\n--- {model_name} Evaluation ---")
    print(classification_report(y_true, y_pred))
    auc = roc_auc_score(y_true, y_prob)
    print(f"AUC: {auc:.2f}")
    return auc

auc_rf = evaluate_model(y_test, y_pred_rf, y_prob_rf, "Random Forest")
auc_gb = evaluate_model(y_test, y_pred_gb, y_prob_gb, "Gradient Boosting")

# Feature Importance Visualization
import matplotlib.pyplot as plt

rf_importances = rf_model.feature_importances_
gb_importances = gb_model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 6))
plt.bar(features, rf_importances, alpha=0.6, label='Random Forest')
plt.bar(features, gb_importances, alpha=0.6, label='Gradient Boosting')
plt.title('Feature Importance Comparison', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Importance Score', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall and ROC Curve Visualization
def plot_curves(y_true, y_prob, model_name):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fpr, tpr, _ = roc_curve(y_true, y_prob)

    plt.figure(figsize=(12, 6))

    # Precision-Recall Curve
    plt.subplot(1, 2, 1)
    plt.plot(recall, precision, label=f'{model_name} Precision-Recall')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve')
    plt.legend()

    # ROC Curve
    plt.subplot(1, 2, 2)
    plt.plot(fpr, tpr, label=f'{model_name} ROC (AUC = {roc_auc_score(y_true, y_prob):.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()

plot_curves(y_test, y_prob_rf, "Random Forest")
plot_curves(y_test, y_prob_gb, "Gradient Boosting")
