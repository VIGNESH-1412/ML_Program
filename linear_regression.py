import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('canada_per_capita_income.csv')
df.head()

df.isnull().sum()

df.describe()

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
plt.xlabel('year')
plt.ylabel('canada_per_capita_income')
plt.scatter(df.year,df['per capita income (US$)'],color='red',marker='*')

x = df.drop('per capita income (US$)',axis='columns')
y = df['per capita income (US$)']
df

reg = linear_model.LinearRegression()
reg.fit(x,y)

accuracy_score = reg.score(x,y)
accuracy_score

reg.predict([[2010]])