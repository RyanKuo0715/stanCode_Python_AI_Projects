"""
File: titanic_level2.py
Name: Ryan Kuo
----------------------------------
This file builds a machine learning algorithm by pandas and sklearn libraries.
We'll be using pandas to read in dataset, store data into a DataFrame,
standardize the data by sklearn, and finally train the model and
test it on kaggle website. Hyper-parameters tuning are not required due to its
high level of abstraction, which makes it easier to use but less flexible.
You should find a good model that surpasses 77% test accuracy on kaggle.
"""

import math
import pandas as pd
from sklearn import preprocessing, linear_model

TRAIN_FILE = 'titanic_data/train.csv'
TEST_FILE = 'titanic_data/test.csv'


def data_preprocess(filename, mode='Train', training_data=None):
	"""
	:param filename: str, the filename to be read into pandas
	:param mode: str, indicating the mode we are using (either Train or Test)
	:param training_data: DataFrame, a 2D data structure that looks like an excel worksheet
						  (You will only use this when mode == 'Test')
	:return: Tuple(data, labels), if the mode is 'Train'; or return data, if the mode is 'Test'
	"""
	data = pd.read_csv(filename)
	labels = None
	data.pop('PassengerId')
	data.pop('Name')
	data.pop('Ticket')
	data.pop('Cabin')

	if mode == 'Train':
		data = data.dropna()
		labels = data.pop('Survived')
	elif mode == 'Test':
		data.Age.fillna(round(training_data.Age.mean(), 3), inplace=True)
		data.Fare.fillna(round(training_data.Fare.mean(), 3), inplace=True)

	data.loc[data.Sex == 'male', 'Sex'] = 1
	data.loc[data.Sex == 'female', 'Sex'] = 0
	data.loc[data.Embarked == 'S', 'Embarked'] = 0
	data.loc[data.Embarked == 'C', 'Embarked'] = 1
	data.loc[data.Embarked == 'Q', 'Embarked'] = 2

	if mode == 'Train':
		return data, labels
	elif mode == 'Test':
		return data


def one_hot_encoding(data, feature):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param feature: str, the column name of interest
	:return data: DataFrame, remove the feature column and add its one-hot encoding features
	"""
	if feature == 'Sex':
		# one hot encoding for Sex
		data['Sex_0'] = 0
		data.loc[data.Sex == 0, 'Sex_0'] = 1
		data['Sex_1'] = 0
		data.loc[data.Sex == 1, 'Sex_1'] = 1
		data.pop('Sex')
	elif feature == 'Pclass':
		# one hot encoding for Pclass
		data['Pclass_0'] = 0
		data.loc[data.Pclass == 1, 'Pclass_0'] = 1
		data['Pclass_1'] = 0
		data.loc[data.Pclass == 2, 'Pclass_1'] = 1
		data['Pclass_2'] = 0
		data.loc[data.Pclass == 3, 'Pclass_2'] = 1
		data.pop('Pclass')
	elif feature == 'Embarked':
		# one hot encoding for Embarked
		data['Embarked_0'] = 0
		data.loc[data.Embarked == 0, 'Embarked_0'] = 1
		data['Embarked_1'] = 0
		data.loc[data.Embarked == 1, 'Embarked_1'] = 1
		data['Embarked_2'] = 0
		data.loc[data.Embarked == 2, 'Embarked_2'] = 1
		data.pop('Embarked')

	return data


def standardization(data, mode='Train'):
	"""
	:param data: DataFrame, key is the column name, value is its data
	:param mode: str, indicating the mode we are using (either Train or Test)
	:return data: DataFrame, standardized features
	"""
	standardizer = preprocessing.StandardScaler()

	if mode == 'Train':
		data = standardizer.fit_transform(data)
	elif mode == 'Test':
		data = standardizer.transform(data)

	return data


def main():
	"""
	You should call data_preprocess(), one_hot_encoding(), and
	standardization() on your training data. You should see ~80% accuracy on degree1;
	~83% on degree2; ~87% on degree3.
	Please write down the accuracy for degree1, 2, and 3 respectively below
	(rounding accuracies to 8 decimal places)
	TODO: real accuracy on degree1 -> 0.80196629
	TODO: real accuracy on degree2 -> 0.83707865
	TODO: real accuracy on degree3 -> 0.87640449
	"""
	data, labels = data_preprocess(TRAIN_FILE, mode='Train')
	data = one_hot_encoding(data, 'Sex')
	data = one_hot_encoding(data, 'Pclass')
	data = one_hot_encoding(data, 'Embarked')
	standardizer = preprocessing.StandardScaler()
	data = standardizer.fit_transform(data)

	poly_phi_extractor2 = preprocessing.PolynomialFeatures(degree=2)
	data_poly2 = poly_phi_extractor2.fit_transform(data)
	poly_phi_extractor3 = preprocessing.PolynomialFeatures(degree=3)
	data_poly3 = poly_phi_extractor3.fit_transform(data)

	h = linear_model.LogisticRegression(max_iter=10000)
	classifier1 = h.fit(data, labels)
	print(f'Training Accuracy (degree 1): {classifier1.score(data, labels)}')
	classifier2 = h.fit(data_poly2, labels)
	print(f'Training Accuracy (degree 2): {classifier2.score(data_poly2, labels)}')
	classifier3 = h.fit(data_poly3, labels)
	print(f'Training Accuracy (degree 3): {classifier3.score(data_poly3, labels)}')


if __name__ == '__main__':
	main()
