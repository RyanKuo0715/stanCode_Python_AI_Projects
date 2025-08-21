"""
File: boston_housing_competition.py
Name: example provided by Jerry
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, metrics

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'


def main():
	train_data = pd.read_csv(TRAIN_FILE)
	test_data = pd.read_csv(TEST_FILE)

	y = train_data.medv
	x_train = train_data.rm
	# h = wx + b
	w, b, c = 0, 0, 0.6
	alpha = 0.01
	num_epoch = 100
	history = []
	for epoch in range(num_epoch):
		total = 0
		for i in range(len(x_train)):
			x = x_train[i]
			label = y[i]
			h = w*x+b
			loss = (h-label)**2
			total += loss
			# G.D.
			# w = w - alpha*(2*(h-label)*x)
			w = w - alpha*(2*(h-label)*x*(sign(h-label)-c)**2)
			# b = b - alpha*(2*(h-label)*1)
			b = b - alpha*(2*(h-label)*1*(sign(h-label)-c)**2)
		history.append(total/len(x_train))
	plt.plot(history)
	plt.show()

	predictions = []
	for x in x_train:
		predictions.append(w*x+b)
	print(sum(predictions)/len(predictions))


def data_preprocess(data, mode):
	labels = None
	data.pop('ID')
	if mode == 'Train':
		labels = data.medv
	data = np.array(data.rm).reshape(-1, 1)

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def sign(data):
	if data > 0:
		return 1
	elif data == 0:
		return 0
	else:
		return -1


if __name__ == '__main__':
	main()
