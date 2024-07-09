#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[51]:


df = pd.read_csv('Amazon_Sale_Report.csv', encoding= 'unicode_escape')


# In[52]:


df.shape


# In[53]:


df.info()


# In[54]:


df.head(10)


# In[55]:


df.drop(['index', 'New', 'PendingS'], axis=1, inplace=True)  #axix=1 = full coloum will delet, inplace=true for saving permenently 


# In[56]:


pd.isnull(df).sum()


# In[57]:


df.dropna(inplace=True)


# In[58]:


df.info()


# In[59]:


df.columns


# In[60]:


df['ship-postal-code']=df['ship-postal-code'].astype('int')


# In[61]:


df['Date']= pd.to_datetime(df['Date']) 


# In[62]:


df.info()


# In[63]:


df.describe() #only for nemurical


# In[64]:


df.describe(include='object')


# In[65]:


df.columns


# In[66]:


sz = sns.countplot(x='Size', data=df)


# # Most people are buying M size

# In[67]:


#sales by month
monthly_sales = df.resample('M', on='Date').sum()['Amount']

import matplotlib.pyplot as plt

#monthly sales trend
plt.figure(figsize=(12, 6))
plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
plt.title('Monthly Sales Trend')
plt.xlabel('Month')
plt.ylabel('Total Sales Amount')
plt.grid(True)
plt.show()



# In[68]:


total_sales = df['Amount'].sum()
average_sales_per_day = df.resample('D', on='Date').sum()['Amount'].mean()
sales_growth_rate = monthly_sales.pct_change().mean() * 100

print(f"Total Sales: {total_sales}")
print(f"Average Sales per Day: {average_sales_per_day}")
print(f"Monthly Sales Growth Rate: {sales_growth_rate:.2f}%")


# # sale is high in the month of may

# In[69]:


#sales by product category
category_sales = df.groupby('Category')['Amount'].sum().sort_values(ascending=False)

# quantities by product category
category_quantity = df.groupby('Category')['Qty'].sum().sort_values(ascending=False)


# In[70]:


# sales by product size
size_sales = df.groupby('Size')['Amount'].sum().sort_values(ascending=False)

# Aggregate quantities by product size
size_quantity = df.groupby('Size')['Qty'].sum().sort_values(ascending=False)


# In[92]:


# Plot sales by product category
plt.figure(figsize=(12, 6))
category_sales.plot(kind='bar')
plt.title('Sales by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Sales Amount')
plt.show()

# Plot quantities by product category
plt.figure(figsize=(12, 6))
category_quantity.plot(kind='bar')
plt.title('Quantities Sold by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Total Quantities Sold')
plt.show()

# Plot sales by product size
plt.figure(figsize=(12, 6))
size_sales.plot(kind='bar')
plt.title('Sales by Product Size')
plt.xlabel('Product Size')
plt.ylabel('Total Sales Amount')
plt.show()

# Plot quantities by product size
plt.figure(figsize=(12, 6))
size_quantity.plot(kind='bar')
plt.title('Quantities Sold by Product Size')
plt.xlabel('Product Size')
plt.ylabel('Total Quantities Sold')
plt.show()


# # T-shirt is the most buy product and size M is most ordered.

# In[73]:


#sales by fulfillment method
fulfillment_sales = df.groupby('Fulfilment')['Amount'].sum().sort_values(ascending=False)

#number of orders by fulfillment method
fulfillment_orders = df['Fulfilment'].value_counts()

#quantities by fulfillment method
fulfillment_quantities = df.groupby('Fulfilment')['Qty'].sum().sort_values(ascending=False)

# Calculate average order value by fulfillment method
fulfillment_avg_order_value = fulfillment_sales / fulfillment_orders


# In[74]:


#courier status by fulfillment method
courier_status = df.groupby(['Fulfilment', 'Courier Status']).size().unstack(fill_value=0)


# In[75]:


#Plot sales by fulfillment method
plt.figure(figsize=(12, 6))
fulfillment_sales.plot(kind='bar')
plt.title('Sales by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Total Sales Amount')
plt.show()

#Plot number of orders by fulfillment method
plt.figure(figsize=(12, 6))
fulfillment_orders.plot(kind='bar')
plt.title('Number of Orders by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Number of Orders')
plt.show()

#Plot quantities by fulfillment method
plt.figure(figsize=(12, 6))
fulfillment_quantities.plot(kind='bar')
plt.title('Quantities Sold by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Total Quantities Sold')
plt.show()

#Plot average order value by fulfillment method
plt.figure(figsize=(12, 6))
fulfillment_avg_order_value.plot(kind='bar')
plt.title('Average Order Value by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Average Order Value')
plt.show()

#Plot courier status distribution by fulfillment method
courier_status.plot(kind='bar', stacked=True, figsize=(12, 6))
plt.title('Courier Status Distribution by Fulfillment Method')
plt.xlabel('Fulfillment Method')
plt.ylabel('Number of Orders')
plt.show()


# 
# # 35000+  number of orders by fulfillment method
# # 30000+ quantities by fulfillment method
# # 600+ is average order value by fulfillment method
# # most of the courier status is been shipped

# In[83]:


#Aggregate customer data by order ID to get total spending, number of orders, and average order value
customer_data = df.groupby('Order ID').agg({
    'Amount': 'sum',
    'Qty': 'sum',
    'ship-city': 'first',
    'ship-state': 'first',
    'ship-country': 'first'
}).reset_index()

customer_data['Average Order Value'] = customer_data['Amount'] / customer_data['Qty']


# In[85]:


#Prepare data for clustering
segmentation_data = customer_data[['Amount', 'Qty', 'Average Order Value']]


# In[86]:


state_segmentation = df['ship-state'].value_counts().reset_index()
state_segmentation.columns = ['State', 'Number of Orders']
custom_palette = sns.color_palette("rocket_r", len(state_segmentation))
plt.figure(figsize=(14, 7))
sns.barplot(data=state_segmentation, x='State', y='Number of Orders',palette=custom_palette)
plt.title('Customer Segmentation by State')
plt.xlabel('State')
plt.ylabel('Number of Orders')
plt.xticks(rotation=90)


# # most of the customers are from maharashtra over 6000+

# In[89]:


geo_sales = df.groupby(['ship-state', 'ship-city']).agg({'Amount': 'sum'}).reset_index()


# In[99]:


state_sales = geo_sales.groupby('ship-state').agg({'Amount': 'sum'}).reset_index()
state_sales = state_sales.sort_values('Amount', ascending=False)
plt.figure(figsize=(14, 7))
sns.barplot(data=state_sales, x='ship-state', y='Amount', palette='rocket_r')
plt.title('Sales by State')
plt.xlabel('State')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()


# # 390000+ sales is in maharsahstra itself

# In[100]:


city_sales = geo_sales.groupby('ship-city').agg({'Amount': 'sum'}).reset_index()
city_sales = city_sales.sort_values('Amount', ascending=False).head(10)
plt.figure(figsize=(14, 7))
sns.barplot(data=city_sales, x='ship-city', y='Amount', palette='rocket_r')
plt.title('Sales by City (Top 10)')
plt.xlabel('City')
plt.ylabel('Total Sales')
plt.xticks(rotation=90)
plt.show()


# # Bengaluru is the top city for highest sales in itself when compared to cities combined in maharshatra (mumbai,pune,thane)
