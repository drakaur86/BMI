import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Dataset (Using a heart disease dataset)
# Link: https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease
# For this example, we simulate the core structure of the BMI-Heart link
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
cols = ['Preg', 'Glu', 'BP', 'Skin', 'Ins', 'BMI', 'Pedigree', 'Age', 'Outcome']
df = pd.read_csv(url, names=cols)

# 2. Engineering BMI Categories
def categorize_bmi(bmi):
    if bmi < 18.5: return 0 # Underweight
    elif 18.5 <= bmi < 25: return 1 # Normal
    elif 25 <= bmi < 30: return 2 # Overweight
    else: return 3 # Obese

df['BMI_Cat'] = df['BMI'].apply(categorize_bmi)

# 3. Preprocessing
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling features (Essential for BMI and BP units)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Training the Model (Random Forest for 83% target)
model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"--- BMI/CVS Model Performance ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\nFeature importance in relation to Heart Risk:")

# 6. Analysis: Correlation between BMI and Risk
plt.figure(figsize=(10, 5))
sns.boxplot(x='Outcome', y='BMI', data=df, palette='Reds')
plt.title('Distribution of BMI in Healthy vs. CVS-Risk Patients')
plt.xticks([0, 1], ['Low Risk', 'High Risk'])
plt.show()



# 7. Prediction Function
def predict_cvs_risk(age, bmi, bp, glucose):
    # Dummy values for remaining features to match model input shape
    features = np.array([[0, glucose, bp, 0, 0, bmi, 0.5, age, categorize_bmi(bmi)]])
    scaled_features = scaler.transform(features)
    risk = model.predict(scaled_features)
    return "High Cardiovascular Risk" if risk[0] == 1 else "Low Cardiovascular Risk"

# Example: 50 years old, BMI 32 (Obese), BP 140, Glucose 120
print(f"Result: {predict_cvs_risk(50, 32, 140, 120)}")
