import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
df = pd.read_csv("titanic.csv")

# Display First 5 Rows
print("\nFirst Five Records")
print(df.head())

# Dataset Information
print("\nDataset Information")
print(df.info())

# Statistical Summary
print("\nStatistical Summary")
print(df.describe())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())

# Fill Missing Age Values
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill Missing Embarked Values
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop Cabin Column
df.drop(columns=["Cabin"], inplace=True)

# Check Missing Values Again
print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# -------------------------------
# Visualization 1: Survival Count
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# -------------------------------
# Visualization 2: Passenger Class
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Passenger Class vs Survival")
plt.savefig("class_survival.png")
plt.show()

# -------------------------------
# Visualization 3: Gender Survival
# -------------------------------
plt.figure(figsize=(6,4))
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Gender vs Survival")
plt.savefig("gender_survival.png")
plt.show()

# -------------------------------
# Visualization 4: Age Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# -------------------------------
# Visualization 5: Fare Distribution
# -------------------------------
plt.figure(figsize=(8,5))
sns.histplot(df["Fare"], bins=30, kde=True)
plt.title("Fare Distribution")
plt.savefig("fare_distribution.png")
plt.show()

# -------------------------------
# Visualization 6: Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))

numeric_df = df.select_dtypes(include=["number"])

sns.heatmap(numeric_df.corr(),
            annot=True,
            cmap="coolwarm")

plt.title("Correlation Heatmap")
plt.savefig("heatmap.png")
plt.show()

print("\nEDA Completed Successfully!")