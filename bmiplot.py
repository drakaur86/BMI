import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# 1. BMI Distribution Plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='CVD_Risk', y='BMI', data=df, palette='Set2')
plt.title('BMI Distribution vs. Cardiovascular Risk Status', fontsize=14)
plt.xlabel('CVD Risk (0: Low, 1: High)', fontsize=12)
plt.ylabel('Body Mass Index (BMI)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig('bmi_vs_cvd_risk.png')

# 2. Feature Importance Plot
importances = model.feature_importances_
features = X.columns
indices = np.argsort(importances)

plt.figure(figsize=(10, 6))
plt.title('Factor Analysis: Impact of BMI vs Other Metrics', fontsize=14)
plt.barh(range(len(indices)), importances[indices], color='salmon', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance Score', fontsize=12)
plt.savefig('bmi_feature_importance.png')

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='OrRd', 
            xticklabels=['Low Risk', 'High Risk'], 
            yticklabels=['Low Risk', 'High Risk'])
plt.title('CVD Prediction Matrix', fontsize=14)
plt.savefig('cvd_confusion_matrix.png')
