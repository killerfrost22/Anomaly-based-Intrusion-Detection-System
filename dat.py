import pandas as pd

df = pd.read_csv('Train_data.csv')

print(df.shape)
df1 = pd.get_dummies(df)
# print(df1.shape)
# df.replace('normal', 1)
# df.replace('anomaly', 0)
print(df1.head(5))