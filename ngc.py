import numpy as np

K = 5 # internal layer
epsilon = 0.01
MaxItr = 100

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

W, v = train(X, Y, MaxItr)
print(W)
print(v)
