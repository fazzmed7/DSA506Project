!pip install -U kaleido
import kaleido
import plotly
print(f"Plotly version: {plotly.__version__}")
print(f"Kaleido version: {kaleido.__version__}")
import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)

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
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error


# Load Dataset
df = pd.read_csv('/workspaces/DSA506Project/Sample - Superstore.csv', encoding='windows-1254')

# Data Cleaning
df.drop_duplicates(inplace=True)
df.dropna(inplace=True)
#Transform data types
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Lead time'] = (df['Ship Date']-df['Order Date']).dt.days
df['year_month'] = df['Order Date'].dt.strftime('%Y-%m')
#Add derived columns
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day

# Exploratory Data Analysis (EDA)
#Total Sales and Profits across regions
#Group data by region and calculate total sales 
regional_sales=df.groupby('Region')['Sales'].sum().reset_index()
#Create an interactive bar plot
fig=px.bar(regional_sales,x='Region',y='Sales',title='Total Sales by Region',
           labels={'Sales':'Total Sales','Region':'Region'},
           color='Region',text='Sales')
fig.update_traces(textposition='outside')
fig.write_image("plot0.png")
fig.show()

# Understanding sales distribution across different product categories.
plt.figure(figsize=(10, 5))
df.groupby('Category')['Sales'].sum().plot(kind='bar', color='skyblue')
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.savefig("plot1.jpg", dpi=300)
plt.show()

# Sales Across Different Locations
# Identifying which states generate the most revenue.
plt.figure(figsize=(12, 6))
df.groupby('State')['Sales'].sum().sort_values(ascending=False).plot(kind='bar', color='teal')
plt.title("Total Sales by State")
plt.xlabel("State")
plt.ylabel("Total Sales")
plt.xticks(rotation=90)
plt.savefig("plot2.jpg", dpi=300)
plt.show()

# Display List of States Contributing to Sales
state_sales_list = df.groupby('State')['Sales'].sum().reset_index()
print("List of States Contributing to Sales:")
print(state_sales_list)

# Display Unique Categorical Values of States
unique_states = df['State'].unique()
print("Unique States:", unique_states)


# Profit Across Different Locations
# Analyzing profitability distribution by state to identify key revenue-driving locations.
plt.figure(figsize=(12, 6))
df.groupby('State')['Profit'].sum().sort_values(ascending=False).plot(kind='bar', color='green')
plt.title("Total Profit by State")
plt.xlabel("State")
plt.ylabel("Total Profit")
plt.xticks(rotation=90)
plt.savefig("plot3.jpg", dpi=300)
plt.show()

# Total Profit Calculation
# Summing up the total profit to gauge overall business performance.
total_profit = df['Profit'].sum()
print(f"Total Profit: ${total_profit:,.2f}")

# Approximate Latitude and Longitude for US States
state_coordinates = {
    "Alabama": [32.806671, -86.791130], "Arizona": [33.729759, -111.431221], "Arkansas": [34.969704, -92.373123],
    "California": [36.116203, -119.681564], "Colorado": [39.059811, -105.311104], "Connecticut": [41.597782, -72.755371],
    "Delaware": [39.318523, -75.507141], "Florida": [27.766279, -81.686783], "Georgia": [33.040619, -83.643074],
    "Idaho": [44.240459, -114.478828], "Illinois": [40.349457, -88.986137], "Indiana": [39.849426, -86.258278],
    "Iowa": [42.011539, -93.210526], "Kansas": [38.526600, -96.726486], "Kentucky": [37.668140, -84.670067],
    "Louisiana": [31.169546, -91.867805], "Maine": [44.693947, -69.381927], "Maryland": [39.063946, -76.802101],
    "Massachusetts": [42.230171, -71.530106], "Michigan": [43.326618, -84.536095], "Minnesota": [45.694454, -93.900192],
    "Mississippi": [32.741646, -89.678696], "Missouri": [38.456085, -92.288368], "Montana": [46.921925, -110.454353],
    "Nebraska": [41.125370, -98.268082], "Nevada": [38.313515, -117.055374], "New Hampshire": [43.452492, -71.563896],
    "New Jersey": [40.298904, -74.521011], "New Mexico": [34.840515, -106.248482], "New York": [42.165726, -74.948051],
    "North Carolina": [35.630066, -79.806419], "North Dakota": [47.528912, -99.784012], "Ohio": [40.388783, -82.764915],
    "Oklahoma": [35.565342, -96.928917], "Oregon": [44.572021, -122.070938], "Pennsylvania": [40.590752, -77.209755],
    "Rhode Island": [41.680893, -71.511780], "South Carolina": [33.856892, -80.945007], "South Dakota": [44.299782, -99.438828],
    "Tennessee": [35.747845, -86.692345], "Texas": [31.054487, -97.563461], "Utah": [40.150032, -111.862434],
    "Vermont": [44.045876, -72.710686], "Virginia": [37.769337, -78.169968], "Washington": [47.400902, -121.490494],
    "West Virginia": [38.491226, -80.954456], "Wisconsin": [44.268543, -89.616508], "Wyoming": [42.755966, -107.302490]
}

# Add Coordinates to DataFrame
state_sales = df.groupby('State', as_index=False)['Sales'].sum()
state_sales['Latitude'] = state_sales['State'].map(lambda x: state_coordinates.get(x, [None, None])[0])
state_sales['Longitude'] = state_sales['State'].map(lambda x: state_coordinates.get(x, [None, None])[1])

# Aggregate Sales and Profit by State
state_sales_profit = df.groupby('State').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Generate Interactive Map with State Coordinates and Sales/Profit Info
def generate_map(df, state_sales_profit):
    store_map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)
    for _, row in state_sales_profit.iterrows():
        state = row['State']
        coords = state_coordinates.get(state, [None, None])
        if coords[0] is not None and coords[1] is not None:
            popup_text = f"State: {state}<br>Sales: ${row['Sales']:,.2f}<br>Profit: ${row['Profit']:,.2f}"
            marker_color = 'green' if row['Profit'] > 0 else 'yellow'
            folium.Marker(
                location=coords,
                popup=popup_text,
                icon=folium.Icon(color=marker_color)
            ).add_to(store_map)
    return store_map

sales_map = generate_map(df, state_sales_profit)
sales_map.save("sales_map.html")


# Correlation Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Sales', 'Profit', 'Quantity', 'Discount']].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.savefig("plot5.jpg", dpi=300)
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
plt.savefig("plot6.jpg", dpi=300)
plt.show()

# Clustering (Customer Segmentation based on Sales & Profit)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Sales', 'Profit']])

plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Sales'], y=df['Profit'], hue=df['Cluster'], palette='viridis')
plt.title("Customer Segmentation Based on Sales & Profit")
plt.xlabel("Sales")
plt.ylabel("Profit")
plt.savefig("plot7.jpg", dpi=300)
plt.show()

# Model Performance Evaluation
X = df[['Sales']]
y = df['Profit']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

df.set_index('Order Date', inplace=True)
y_test = df['Profit'].iloc[-len(y_test):]  

 # Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
lr_accuracy = r2_score(y_test, y_pred_lr)

# Time Series SARIMA Model
# Time Series SARIMA Model
ts_model = SARIMAX(df['Profit'], order=(1,1,1), seasonal_order=(1,1,1,12))
ts_result = ts_model.fit()
y_pred_sarima = ts_result.predict(start=len(y_train), end=len(y_train) + len(y_test) - 1)

# Ensure alignment between predictions and actual values
y_pred_sarima = y_pred_sarima[:len(y_test)] 
ts_mape = mean_absolute_percentage_error(y_test, y_pred_sarima)
ts_accuracy = 1 - ts_mape 

# KMeans Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)
if len(set(kmeans.labels_)) > 1:  # Ensure multiple clusters exist
    kmeans_accuracy = silhouette_score(X, kmeans.labels_)
else:
    kmeans_accuracy = 0

print("Linear Regression Accuracy:", lr_accuracy)
print("SARIMA (Time Series) Accuracy:", ts_accuracy)
print("KMeans Clustering Accuracy:", kmeans_accuracy)

# Model Performance Summary Graph
models = ['Linear Regression', 'SARIMA (Time Series)', 'KMeans Clustering']
accuracies = [lr_accuracy, ts_accuracy, kmeans_accuracy]

plt.figure(figsize=(8,5))
plt.bar(models, accuracies, color=['blue', 'green', 'red'])
plt.xlabel("Model")
plt.ylabel("Accuracy Score")
plt.title("Comparison of Model Performance")
plt.ylim(-1, 1) 
plt.savefig("plot8.jpg", dpi=300)
plt.show()

#Monthly sales by category
monthly_category_sales = df.groupby(['Month', 'Category'])['Sales'].sum().reset_index()
plt.figure(figsize=(12, 6))
for category in monthly_category_sales['Category'].unique():
    data = monthly_category_sales[monthly_category_sales['Category'] == category]
    plt.plot(data['Month'], data['Sales'], label=category)
plt.title('Monthly Sales Trends by Category')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend()
plt.savefig('plot7.png')
plt.show()
sns.scatterplot(x='Sales', y='Profit', hue='Category', data=df)
plt.title('Sales vs. Profit by Category')
plt.savefig('plot8.png')
plt.show()

# Group data by category and calculate average sales, profit, and discount
category_metrics = df.groupby('Category').agg({'Sales': 'mean', 'Profit': 'mean', 'Discount': 'mean'}).reset_index()
category_metrics


#Trends & Patterns over time
#Group by month & calculate total sales & profit

df['YearMonth'] = df['Order Date'].dt.strftime('%Y-%m')
monthly_data = df.groupby('YearMonth').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
plt.figure(figsize=(12, 6))
plt.plot(monthly_data['YearMonth'], monthly_data['Sales'], marker='o')
plt.title('Monthly Sales Trends')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.grid()
plt.savefig('plot10.png')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(monthly_data['YearMonth'], monthly_data['Profit'], marker='o', color='orange')
plt.title('Monthly Profit Trends')
plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.grid()
plt.savefig('plot11.png')
plt.show()

# Perform seasonal decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
# Set the 'YearMonth' column as the index
monthly_data.set_index('YearMonth', inplace=True)

decomposition = seasonal_decompose(monthly_data['Sales'], model='additive', period=12)
decomposition.plot()
plt.savefig('plot12.png')
plt.show()

#Forecasting future sales
train = monthly_data.iloc[:24]  # First 24 months for training
test = monthly_data.iloc[24:]   # Remaining months for testing
from statsmodels.tsa.arima.model import ARIMA
# Fit the ARIMA model
model = ARIMA(train['Sales'], order=(5, 1, 0))  # Example order (p, d, q)
model_fit = model.fit()

# Forecast future sales
forecast = model_fit.forecast(steps=len(test))
plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Sales'], label='Training Data')
plt.plot(test.index, test['Sales'], label='Actual Sales')
plt.plot(test.index, forecast, label='Forecasted Sales', color='red')
plt.title('Sales Forecast vs. Actual')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.xticks(rotation=45)
plt.legend()
plt.grid()
plt.savefig('plot13.png')
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
