import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df =pd.read_csv("churn.csv")
print("Shape",df.shape)
print("/Churn counts:\n",df['Churn'].value_counts())

# Fix TotalCharges column
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(inplace=True)

# Plot Churn Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='Churn',data=df, hue='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title("Churn Distrubution")
plt.xlabel('Churn')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('data/churn_distribution.png')
plt.show()

# Churn by Contract Type
plt.figure(figsize=(8,5))
sns.countplot(x='Contract', hue='Churn', data=df, palette=['#2ecc71', '#e74c3c'])
plt.title('Churn by Contract Type')
plt.tight_layout()
plt.savefig('data/churn_by_contract.png')
plt.show()

# Monthly Charges vs Churn
plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df, hue='Churn', palette=['#2ecc71', '#e74c3c'])
plt.title('Monthly Charges vs Churn')
plt.tight_layout()
plt.savefig('data/monthly_charges.png')
plt.show()

# Tenure vs Churn
plt.figure(figsize=(8,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, palette=['#2ecc71', '#e74c3c'])
plt.title('Tenure vs Churn')
plt.tight_layout()
plt.savefig('data/tenure_churn.png')
plt.show()

print("\n✅ EDA complete! Charts saved in data/ folder")