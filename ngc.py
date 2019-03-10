import numpy as np
import pandas as pd

K = 10 # internal layer
epsilon = 0.01
MaxItr = 10

path = 'Train_data_cleaned-2.0.2.csv'

df = pd.read_csv(path)
Y = df.values[:, -1]
df = df.drop(columns=['class'])
data = pd.get_dummies(df)
X = data.values

N, D = X.shape
num = int(N * 0.8)
X_train = X[0:num, :]
Y_train = Y[0:num]
X_val = X[num:, :]
Y_val = Y[num:]

def train(X, Y, MaxItr):
	N, D = X.shape
	W = np.random.random_sample((D, K))
	v = np.random.random_sample(K)
	for i in range(MaxItr):
		G = np.zeros((D, K))
		g = np.zeros(K)
		a = np.zeros(K)
		h = np.zeros(K)
		for j in range(N):
			for k in range(K):
				a[k] = np.dot(W[:,k], X[j])
				h[k] = np.tanh(a[k])
			y = np.dot(v, h)
			e = y - Y[j]
			g = g - e * h
			for k in range(K):
				G[:,k] = G[:,k] - e * v[k] * (1 - np.tanh(a[k]) ** 2) * X[j]
		W = W - epsilon * G
		v = v - epsilon * g
	return W, v

def test(X, Y, W, v):
	res = np.matmul(np.tanh(np.matmul(X, W)), v)
	print(res.shape)
	print(res)
	print('--------')
	print(np.amax(res))
	print(np.amin(res))
	print(Y.shape)
	return

W, v = train(X_train, Y_train, MaxItr)
print(W.shape)
print(v.shape)
print(X_val.shape)
print("---------")
test(X_val, Y_val, W, v)
