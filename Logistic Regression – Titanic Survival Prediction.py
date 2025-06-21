import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, log_loss,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay

# Load dataset from seaborn
df = sns.load_dataset("titanic").dropna(subset=["age", "embarked", "sex"])

# Preprocess
df['sex'] = LabelEncoder().fit_transform(df['sex'])  # male=1, female=0
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])  # C=0, Q=1, S=2
X = df[["pclass", "age", "sex", "sibsp", "parch", "fare", "embarked"]]
y = df["survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# --------------------
# 📊 Classification Metrics
# --------------------
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
lloss = log_loss(y_test, y_prob)
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("F1 Score:", round(f1, 3))
print("ROC AUC:", round(auc, 3))
print("Log Loss:", round(lloss, 3))
print("Cohen’s Kappa:", round(kappa, 3))
print("MCC:", round(mcc, 3))
print("Specificity:", round(specificity, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# --------------------
# 📈 Visualizations
# --------------------

# Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve")
plt.show()

# Precision-Recall Curve
PrecisionRecallDisplay.from_estimator(model, X_test, y_test)
plt.title("Precision-Recall Curve")
plt.show()

# Bar Chart of metrics
metrics = [acc, prec, rec, f1, auc, specificity]
labels = ["Accuracy", "Precision", "Recall", "F1", "AUC", "Specificity"]
plt.bar(labels, metrics, color='skyblue')
plt.title("Evaluation Metrics")
plt.ylim(0, 1)
plt.show()

# Pie chart of survived vs not
survived_counts = y.value_counts()
plt.pie(survived_counts, labels=["Not Survived", "Survived"], autopct="%1.1f%%", startangle=90, colors=["red", "green"])
plt.title("Survival Distribution")
plt.axis("equal")
plt.show()

# KDE Plot of age distribution
sns.kdeplot(data=df, x="age", hue="survived", fill=True)
plt.title("Age Distribution by Survival")
plt.show()
