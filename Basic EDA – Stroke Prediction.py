import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

url = "https://github.com/karavokyrismichail/Stroke-Prediction---Random-Forest/raw/main/healthcare-dataset-stroke-data/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url)

categorical = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
for col in categorical:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

df["bmi"].fillna(df["bmi"].median(), inplace=True)

# Visualizations
sns.countplot(data=df, x="stroke")
plt.title("Target Class Distribution")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

for col in ["age", "avg_glucose_level", "bmi"]:
    sns.kdeplot(data=df, x=col, hue="stroke", fill=True)
    plt.title(f"KDE Plot: {col}")
    plt.show()

sns.boxplot(x="stroke", y="age", data=df)
plt.title("Age Distribution by Stroke")
plt.show()

sns.pairplot(df.sample(min(300, len(df))), hue="stroke")
plt.show()
