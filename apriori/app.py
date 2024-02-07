import streamlit as st
import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

myretaildata = pd.read_csv('D:\\internship\\project in regression\\apriori\\OnlineRetail.csv', encoding='latin1')

st.title('Original Retail Data')
st.write(myretaildata.head())

st.title('Cleaned Retail Data')

myretaildata['Description'] = myretaildata['Description'].str.strip()
myretaildata.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
myretaildata['InvoiceNo'] = myretaildata['InvoiceNo'].astype('str')
myretaildata = myretaildata[~myretaildata['InvoiceNo'].str.contains('C')]

st.write(myretaildata.head())

st.title('Country Counts')
st.write(myretaildata['Country'].value_counts())

st.title('Transaction Basket')

mybasket = (myretaildata[myretaildata['Country'] == "Germany"]
            .groupby(['InvoiceNo', 'Description'])['Quantity']
            .sum().unstack().reset_index().fillna(0)
            .set_index('InvoiceNo'))

st.write(mybasket.head())

my_basket_sets = mybasket.applymap(lambda x: 1 if x >= 1 else 0)
my_basket_sets.drop('POSTAGE', inplace=True, axis=1)  # Remove "postage" as an item
my_basket_sets.to_csv('my_basket_sets.csv', index=False)

st.title('Association Rules')

my_basket_sets = pd.read_csv('my_basket_sets.csv')

my_frequent_itemsets = apriori(my_basket_sets, min_support=0.07, use_colnames=True)

my_rules = association_rules(my_frequent_itemsets, metric="lift", min_threshold=1)

st.title('Top 100 Association Rules:')
st.write(my_rules.head(100))

st.title('Count for "ROUND SNACK BOXES SET OF 4 WOODLAND":')
st.write(my_basket_sets['ROUND SNACK BOXES SET OF4 WOODLAND'].sum())

st.title('Count for "SPACEBOY LUNCH BOX":')
st.write(my_basket_sets['SPACEBOY LUNCH BOX'].sum())

filtered_rules = my_rules[(my_rules['lift'] >= 3) & (my_rules['confidence'] >= 0.3)]
st.title('Filtered Association Rules:')
st.write(filtered_rules)

