import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('salaries-2023.csv')

print(df.head())
print(df.shape)
df.info()
print(df.describe())

allowed_languages = ['php', 'js', '.net', 'java']
df = df[df['language'].isin(allowed_languages)]

vilnius_names = ['Vilniuj', 'Vilniua', 'VILNIUJE', 'VILNIUS', 'vilnius', 'Vilniuje']
condition = df['city'].isin(vilnius_names)
df.loc[condition, 'city'] = 'Vilnius'

kaunas_names = ['KAUNAS', 'kaunas', 'Kaune']
condition = df['city'].isin(kaunas_names)
df.loc[condition, 'city'] = 'Kaunas'

print(df.city.value_counts())

allowed_cities = ['Vilnius', 'Kaunas']
df = df[df['city'].isin(allowed_cities)]
print(df.shape)

df_sorted = df.sort_values(by='salary', ascending=False)
print(df_sorted.head(20))

x = df.iloc[:, -2:-1]
y = df.iloc[:, -1].values
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.scatter(x, y)
plt.show()

df = df[df['salary'] <= 6000]
print(df.shape)

x = df.iloc[:, -2:-1]
y = df.iloc[:, -1].values
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.scatter(x, y)
plt.show()

one_hot = pd.get_dummies(df['language'], prefix='lang')
df = df.join(one_hot)
df = df.drop('language', axis=1)

one_hot = pd.get_dummies(df['city'], prefix='city')
df = df.join(one_hot)
df = df.drop('city', axis=1)

print(df.head(10))

sns.heatmap(df.corr(), annot=True)
plt.show()

x = df.iloc[:, 0:2].values  # we take only years and level
y = df.iloc[:, 2].values  # we take the salary
print(x[0:5])

print(y[0:5])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train.shape)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
print(y_pred)

salaries = model.predict([[3, 12], [1, 1]])
print(salaries)

r2 = r2_score(y_test, y_pred)

print(f"R2 Score: {r2} ({r2:.2%})")