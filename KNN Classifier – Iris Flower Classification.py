import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    cohen_kappa_score, matthews_corrcoef
)
from sklearn.decomposition import PCA

# Load dataset
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df["target_name"] = df["target"].map({0: "setosa", 1: "versicolor", 2: "virginica"})

# Preprocess
X = df.drop(columns=["target", "target_name"])
y = df["target"]
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Train
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
kappa = cohen_kappa_score(y_test, y_pred)
mcc = matthews_corrcoef(y_test, y_pred)

print("Accuracy:", round(acc, 3))
print("Precision:", round(prec, 3))
print("Recall:", round(rec, 3))
print("F1 Score:", round(f1, 3))
print("Cohen’s Kappa:", round(kappa, 3))
print("MCC:", round(mcc, 3))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Visuals

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# PCA Plot
pca = PCA(n_components=2)
X_2D = pca.fit_transform(X)
df["PCA1"] = X_2D[:, 0]
df["PCA2"] = X_2D[:, 1]
sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="target_name", palette="Set2")
plt.title("2D PCA Visualization of Iris")
plt.show()

# Pairplot
sns.pairplot(df, hue="target_name")
plt.show()
