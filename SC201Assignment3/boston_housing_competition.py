"""
File: boston_housing_competition.py
Name: Ryan Kuo
--------------------------------
This file demonstrates how to analyze boston
housing dataset. Students will upload their 
results to kaggle.com and compete with people
in class!

You are allowed to use pandas, sklearn, or build the
model from scratch! Go data scientists!
"""

# error for basic model: 4.728838336310968
# competition: https://www.kaggle.com/competitions/sc201-mar2024/submissions

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, linear_model, metrics, model_selection, decomposition

TRAIN_FILE = 'boston_housing/train.csv'
TEST_FILE = 'boston_housing/test.csv'


def main():
	data = pd.read_csv(TRAIN_FILE)
	train_data, val_data = model_selection.train_test_split(data, test_size=0.4, random_state=100)
	test_data = pd.read_csv(TEST_FILE)

	# print(train_data.count())  # no missing data
	# print(val_data.count())  # no missing data
	# print(test_data.count())  # no missing data

	train_data, train_y = data_preprocess(train_data, mode='Train')
	val_data, val_y = data_preprocess(val_data, mode='Train')
	test_data, id_numbers = data_preprocess(test_data, mode='Test')

	standardizer = preprocessing.StandardScaler()
	train_data = standardizer.fit_transform(train_data)
	val_data = standardizer.transform(val_data)
	test_data = standardizer.transform(test_data)

	pca = decomposition.PCA(n_components=3)
	train_data = pca.fit_transform(train_data)
	val_data = pca.transform(val_data)
	test_data = pca.transform(test_data)
	var_retained = sum(pca.explained_variance_ratio_)
	print('Var Retained:', var_retained)

	poly_phi_extractor = preprocessing.PolynomialFeatures(degree=2)
	train_data = poly_phi_extractor.fit_transform(train_data)
	val_data = poly_phi_extractor.transform(val_data)
	test_data = poly_phi_extractor.transform(test_data)

	h = linear_model.LinearRegression()
	predictor = h.fit(train_data, train_y)
	train_predictions = predictor.predict(train_data)
	val_predictions = predictor.predict(val_data)
	test_predictions = predictor.predict(test_data)

	print(f'Training RMS Error: {metrics.mean_squared_error(train_y, train_predictions, squared=False)}')
	print(f'Validating RMS Error: {metrics.mean_squared_error(val_y, val_predictions, squared=False)}')
	print('Test Predictions:', test_predictions)
	out_file(test_predictions, id_numbers, 'submission_Ryan_Kuo.csv')


def data_preprocess(data, mode):
	# items = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptratio', 'black', 'lstat']

	labels = id_numbers = None
	data.pop('zn')
	data.pop('indus')
	data.pop('chas')
	data.pop('nox')
	data.pop('rad')
	data.pop('tax')
	data['rm'] = data['rm']**2
	data['ptratio'] = data['ptratio']**2
	data['black'] = data['black']**0.2
	data['lstat'] = data['lstat']**0.2
	if mode == 'Train':
		data.pop('ID')
		labels = data.pop('medv')
	elif mode == 'Test':
		id_numbers = data.pop('ID')

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data, id_numbers


def out_file(predictions, id_numbers, filename):
	with open(filename, 'w') as out:
		out.write('ID,medv\n')
		for prediction, id_number in zip(predictions, id_numbers):
			out.write(f'{id_number},{prediction}\n')


if __name__ == '__main__':
	main()
