import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
# Load Dataset

df = pd.read_csv('/content/Sample - Superstore.csv', encoding='windows-1254')

# Drop rows with missing value
print(df.isnull().sum())
df.dropna(inplace=True)
#Remove duplicates  
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
fig.show()
#Sales and Profits by Product category & subcategory
category_analysis = df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
category_analysis = df.groupby(['Category', 'Sub-Category']).agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
#total sales by category
sales_by_category = df.groupby('Category')['Sales'].sum().reset_index()
print(sales_by_category)
#total profit by category
profit_by_category = df.groupby('Category')['Profit'].sum().reset_index()
print(profit_by_category)
#sales by category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Sales', data=sales_by_category, palette='viridis')
plt.title('Total Sales by Category')
plt.show()
#Profit by category
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Profit', data=profit_by_category, palette='viridis')
plt.title('Total Profit by Category')
plt.show()
#total sales by subcategory
sales_by_subcategory = df.groupby('Sub-Category')['Sales'].sum().reset_index()
print(sales_by_subcategory)
#total profit by subcategory
profit_by_subcategory = df.groupby('Sub-Category')['Profit'].sum().reset_index()
print(profit_by_subcategory)
#sales by subcategory
plt.figure(figsize=(12, 6))
sns.barplot(x='Sub-Category', y='Sales', data=sales_by_subcategory, palette='viridis')
plt.title('Total Sales by Subcategory')
plt.xticks(rotation=45)
plt.show()
#profit by subcategory
plt.figure(figsize=(12, 6))
sns.barplot(x='Sub-Category', y='Profit', data=profit_by_subcategory, palette='viridis')
plt.title('Total Profit by Subcategory')
plt.xticks(rotation=45)
plt.show()
fig = px.bar(category_analysis, x='Category', y='Sales', color='Sub-Category', 
             title='Sales by Category and Subcategory')
fig.show()
#Trends over time
#Month & Order from column
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Month'] = df['Order Date'].dt.strftime('%Y-%m')
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
plt.show()
sns.scatterplot(x='Sales', y='Profit', hue='Category', data=df)
plt.title('Sales vs. Profit by Category')
plt.show()
# Group data by category and calculate average sales, profit, and discount
category_metrics = df.groupby('Category').agg({'Sales': 'mean', 'Profit': 'mean', 'Discount': 'mean'}).reset_index()

# Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(category_metrics.set_index('Category'), annot=True, cmap='coolwarm')
plt.title('Average Sales, Profit, and Discount by Category')
plt.show()
#Trends & Patterns over time
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Day'] = df['Order Date'].dt.day
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
plt.show()
plt.figure(figsize=(12, 6))
plt.plot(monthly_data['YearMonth'], monthly_data['Profit'], marker='o', color='orange')
plt.title('Monthly Profit Trends')
plt.xlabel('Month')
plt.ylabel('Total Profit')
plt.xticks(rotation=45)
plt.grid()
plt.show()
from statsmodels.tsa.seasonal import seasonal_decompose
# Set the 'YearMonth' column as the index
monthly_data.set_index('YearMonth', inplace=True)

# Perform seasonal decomposition
decomposition = seasonal_decompose(monthly_data['Sales'], model='additive', period=12)
decomposition.plot()
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
plt.show()

