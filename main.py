import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import shap
import numpy as np

# Load Data
df = pd.read_csv("HR-Employee-Attrition.csv")  # Replace with your file name

# Check and drop missing values
print("Missing Values:\n", df.isnull().sum())
df = df.dropna()

# Drop irrelevant columns
df = df.drop(['EmployeeNumber', 'Over18', 'StandardHours', 'EmployeeCount'], axis=1, errors='ignore')

# Drop duplicates
df = df.drop_duplicates()

# Ensure object types are strings
for col in df.select_dtypes(include=['object']):
    df[col] = df[col].astype(str)

# Encode target variable
df['Attrition'] = LabelEncoder().fit_transform(df['Attrition'])  # Yes=1, No=0

# One-hot encode categorical features
df = pd.get_dummies(df)

# Split Data
X = df.drop('Attrition', axis=1)
y = df['Attrition']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale Feature
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Model
model = LogisticRegression(max_iter=3000)
model.fit(X_train_scaled, y_train)

# Evaluate Model
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", round(accuracy * 100, 2), "%")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Attrition", "Attrition"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# SHAP Model Analysis
explainer = shap.Explainer(model, X_train_scaled, feature_names=X.columns)
shap_values = explainer(X_test_scaled)

# Global Feature Importance
shap.summary_plot(shap_values, X_test, plot_type="bar", show=True)

# Local Explanation (1 sample)
shap.plots.force(shap_values[0])  # View in Jupyter or compatible viewer
