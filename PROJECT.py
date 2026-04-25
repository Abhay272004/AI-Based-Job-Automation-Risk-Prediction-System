"""
============================================================
 AI-BASED JOB AUTOMATION RISK PREDICTION SYSTEM
============================================================
"""

import warnings
warnings.filterwarnings("ignore")

import os  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



# ================================
# 🎯 PROJECT INTRODUCTION
# ================================
print("\n📌 Problem Statement:")
print("Predict automation risk of jobs using AI and ML\n")

print("💡 Proposed Solution:")
print("Using Machine Learning to classify job risk levels\n")

print("🎯 Objectives:")
print("✔ Data collection from employment/job sector dataset")
print("✔ Data preprocessing")
print("✔ Data analysis and automation trend study")
print("✔ Machine learning model development")
print("✔ Job risk classification")
print("✔ Risk prediction system")
print("✔ Providing insights on future job safety\n")

print("🛠️ Tools Used:")
print("Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn\n")

# ================================
# 🎨 LIGHT THEME
# ================================
plt.style.use("default")
sns.set_theme(style="whitegrid")

# ================================
# 📂 LOAD DATA
# ================================
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "ilo_ai_risk_predictor.csv")

if not os.path.exists(DATA_PATH):
    DATA_PATH = "ilo_ai_risk_predictor.csv"

print(f"📂 Loading dataset from: {DATA_PATH}\n")
df = pd.read_csv(DATA_PATH)
print(f"   Shape: {df.shape}")

# ================================
# 🧹 DATA PREPROCESSING
# ================================
print("\n🧹 Data Cleaning...")

df.drop_duplicates(inplace=True)

print("Missing Values:\n", df.isnull().sum())

# ================================
# 📊 EDA VISUALIZATIONS
# ================================

# 1. Risk Distribution
plt.figure(figsize=(8,5))
sns.countplot(x="risk_label", data=df, palette=["green","yellow","orange","red"])
plt.title("Risk Distribution")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# 2. GenAI Score Distribution
plt.figure(figsize=(8,5))
sns.histplot(df["genai_automation_score"], bins=20, kde=True, color="skyblue")
plt.title("GenAI Automation Score Distribution")
plt.tight_layout()
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# 4. Boxplot
plt.figure(figsize=(8,5))
sns.boxplot(x="risk_label", y="ai_displacement_risk_score",
            data=df, palette=["green","yellow","orange","red"])
plt.title("Risk Score by Risk Level")
plt.tight_layout()
plt.show()

# 5. Top 10 High Risk Jobs
top_jobs = df.groupby("occupation_label")["ai_displacement_risk_score"] \
             .mean().sort_values(ascending=False).head(10)

plt.figure(figsize=(10,6))
top_jobs.plot(kind="barh", color="red")
plt.title("Top 10 High Risk Jobs")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 6. AI Risk Trend Over Years
yearly = df.groupby("year")["ai_displacement_risk_score"].mean()

plt.figure(figsize=(8,5))
plt.plot(yearly.index, yearly.values, marker="o")
plt.title("AI Risk Trend Over Years")
plt.xlabel("Year")
plt.ylabel("Risk Score")
plt.grid(True)
plt.tight_layout()
plt.show()




# ================================
# 📊 STATISTICAL ANALYSIS
# ================================
print("\n📊 Statistical Analysis:")

print("\nMean Values:\n", df.mean(numeric_only=True))

print("\nCorrelation with GenAI Score:\n",
      df.corr(numeric_only=True)["genai_automation_score"])

# ================================
# 📊 HYPOTHESIS TESTING
# ================================

print("\n📊 T-Test (Low vs High Risk)")

low = df[df["risk_label"] == "Low"]["ai_displacement_risk_score"]
high = df[df["risk_label"] == "High"]["ai_displacement_risk_score"]

t_stat, p_val = stats.ttest_ind(low, high)

print("T-statistic:", t_stat)
print("P-value:", p_val)

if p_val < 0.05:
    print("✅ Significant difference")
else:
    print("❌ No significant difference")

print("\n📊 Chi-Square Test (Risk vs Income Group)")

contingency = pd.crosstab(df["risk_label"], df["income_group"])

chi2, p, dof, expected = stats.chi2_contingency(contingency)

print("Chi-square:", chi2)
print("P-value:", p)

if p < 0.05:
    print("✅ Significant relationship")
else:
    print("❌ No relationship")

# ================================
# 🤖 MACHINE LEARNING
# ================================
print("\n🤖 Training Model...\n")

features = [
    "occupation_label",
    "genai_automation_score",
    "tasks_automatable_of_10",
    "digital_infra_index"
]

X = df[features]
y = df["risk_label"]

le_job = LabelEncoder()
X["occupation_label"] = le_job.fit_transform(X["occupation_label"])

le_target = LabelEncoder()
y = le_target.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=120, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred)*100)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(7,5))
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le_target.classes_,
            yticklabels=le_target.classes_)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# ================================
# 🎯 PREDICTION SYSTEM
# ================================
import difflib

def predict_job(job_name, genai, tasks, digital):
    job_list = df["occupation_label"].tolist()

    match = difflib.get_close_matches(job_name, job_list, n=1, cutoff=0.4)

    if not match:
        return "❌ Job not found"

    real_job = match[0]
    job_encoded = le_job.transform([real_job])[0]

    data = np.array([[job_encoded, genai, tasks, digital]])
    pred = model.predict(data)[0]

    result = le_target.inverse_transform([pred])[0]

    if result.lower() == "low":
        message = "🟢 Safe career — low risk of automation\n💡 Keep upgrading advanced skills"
    elif result.lower() == "medium":
        message = "🟡 Moderate risk — upskilling recommended\n💡 Learn AI & digital tools"
    elif result.lower() == "high":
        message = "🟠 High risk — automation likely\n💡 Shift towards technical/creative roles"
    elif result.lower() == "very high":
        message = "🔴 Critical risk — urgent reskilling needed\n💡 Consider switching to AI-proof careers"
    else:
        message = result

    return f"{real_job} → {message}"

# ================================
# EXAMPLES OF JOBS ROLE
# ================================

job_roles = [
    "Handicraft & Printing Workers",
    "Skilled Agricultural Workers",
    "Machine Operators",
    "Clerical Support Workers",
    "Sales Workers",
    "Technicians",
    "Professionals",
    "Managers",
    "Service Workers",
    "Elementary Occupations"
]

print("\n📌 Sample Job Roles:\n")

for i, job in enumerate(job_roles, 1):
    print(f"{i}. {job}")

# ================================
# 🧪 USER INPUT
# ================================
print("\n🎯 Try Your Own Prediction (type 'exit' to quit)\n")

while True:
    try:
        job = input("Enter Job Name: ")

        if job.lower() == "exit":
            print("👋 Exiting program...")
            break

        genai = float(input("GenAI Score (0-1): "))
        tasks = float(input("Tasks Automatable (0-10): "))
        digital = float(input("Digital Infra (0-100): "))

        result = predict_job(job, genai, tasks, digital)

        print(f"\n🔮 Prediction: {result}\n")

    except:
        print("❌ Invalid Input")
