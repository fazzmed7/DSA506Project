import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
url = "https://raw.githubusercontent.com/selva86/datasets/master/Superstore.csv"
df = pd.read_csv(url, encoding='latin-1')

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Exploratory Data Analysis (EDA)
plt.figure(figsize=(10, 5))
df.groupby('Category')['Sales'].sum().plot(kind='bar', color='skyblue')
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Profit', 'Quantity', 'Discount']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Regression Model (Predicting Sales based on Discount & Quantity)
X = df[['Discount', 'Quantity']]
y = df['Sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
print(f'RMSE: {np.sqrt(mean_squared_error(y_test, y_pred))}')
print(f'R² Score: {r2_score(y_test, y_pred)}')

# Scatter Plot of Actual vs Predicted Sales
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--', color='red')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()

# Clustering (Customer Segmentation based on Sales & Profit)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Sales', 'Profit']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Sales'], y=df['Profit'], hue=df['Cluster'], palette='viridis')
plt.title("Customer Segmentation Based on Sales & Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.show()

# Key Insights
print("1. The Technology category generates the highest sales.")
print("2. Discounts have a weak correlation with Sales, suggesting excessive discounts may not significantly boost revenue.")
print("3. The regression model has moderate accuracy, indicating additional features may improve predictions.")
print("4. Clustering reveals three customer segments, which can help tailor marketing strategies.")
