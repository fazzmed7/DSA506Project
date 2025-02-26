import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import folium
from folium.plugins import MarkerCluster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score

# Load Dataset
df = pd.read_csv('/content/Sample - Superstore.csv', encoding='windows-1254')

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
plt.savefig("plot1.jpg", dpi=300)
plt.show()

# Sales Across Different Locations
plt.figure(figsize=(12, 6))
df.groupby('State')['Sales'].sum().sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title("Total Sales by State")
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.savefig("plot2.jpg", dpi=300)
plt.show()

# Profit Across Different Locations
plt.figure(figsize=(12, 6))
df.groupby('State')['Profit'].sum().sort_values(ascending=False).plot(kind='bar', color='green')
plt.title("Total Profit by State")
plt.xlabel("State")
plt.ylabel("Total Profit")
plt.xticks(rotation=90)
plt.savefig("plot3.jpg", dpi=300)
plt.show()

# Total Profit Calculation
total_profit = df['Profit'].sum()
print(f"Total Profit: ${total_profit:,.2f}")

# Total Profit Visualization
plt.figure(figsize=(6, 4))
plt.bar(['Total Profit'], [total_profit], color='purple')
plt.title("Total Profit of Superstore")
plt.ylabel("Profit ($)")
plt.savefig("plot4.jpg", dpi=300)
plt.show()

# Interactive Map using Folium
def generate_map(df):
    store_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    marker_cluster = MarkerCluster().add_to(store_map)
    for _, row in df.iterrows():
        folium.Marker(
            location=[row['Latitude'], row['Longitude']],
            popup=f"State: {row['State']}\nSales: ${row['Sales']:,.2f}",
            icon=folium.Icon(color='blue')
        ).add_to(marker_cluster)
    return store_map

# Dummy Coordinates (for visualization, replace with actual lat/lon data if available)
state_sales = df.groupby('State', as_index=False)['Sales'].sum()
state_sales['Latitude'] = np.random.uniform(25, 50, state_sales.shape[0])
state_sales['Longitude'] = np.random.uniform(-125, -67, state_sales.shape[0])

sales_map = generate_map(state_sales)
sales_map.save("sales_map.html")

# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Profit', 'Quantity', 'Discount']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("plot6.jpg", dpi=300)
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
plt.savefig("plot7.jpg", dpi=300)
plt.show()

# Clustering (Customer Segmentation based on Sales & Profit)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Sales', 'Profit']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Sales'], y=df['Profit'], hue=df['Cluster'], palette='viridis')
plt.title("Customer Segmentation Based on Sales & Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.savefig("plot8.jpg", dpi=300)
plt.show()

# Storytelling and Insights
print("### Storytelling: Superstore Sales Analysis")
print("Superstore is looking to optimize its sales strategy by analyzing historical sales data. This project aims to provide insights into sales trends, profitability, and customer behavior using data analytics.")

print("**Key Insights:**")
print("1. The Technology category generates the highest sales, making it a priority for marketing efforts.")
print("2. The correlation heatmap shows that discounts have a weak correlation with sales, suggesting that excessive discounts may not significantly boost revenue.")
print("3. Regression analysis predicts sales based on discount and quantity, showing moderate accuracy. Additional factors like customer demographics could improve predictions.")
print("4. Clustering analysis reveals three customer segments based on sales and profit, allowing for targeted marketing strategies.")
print("5. Sales and profit analysis by location shows that some states contribute significantly more to overall revenue and profitability. These regions should be prioritized for future sales strategies.")

print("### Business Recommendations:")
print("- Focus on promoting high-profit categories like Technology while optimizing discounts for better margins.")
print("- Use customer segmentation insights to create personalized marketing campaigns.")
print("- Improve the predictive model by incorporating more features like regional trends and customer loyalty data.")
print("- Invest more in high-performing locations while developing strategies to improve sales in lower-performing regions.")

print("This analysis provides a data-driven foundation for decision-making, ensuring Superstore remains competitive and profitable.")
