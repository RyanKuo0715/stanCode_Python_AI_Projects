"""
File: validEmailAddress_2.py
Name: Ryan Kuo
----------------------------
Please construct your own feature vectors
and try to surpass the accuracy achieved by
Jerry's feature vector in validEmailAddress.py.
feature1: TODO: 1 and just only 1 '@' in string
feature2: TODO: some strings before '@'
feature3: TODO: some strings after '@'
feature4: TODO: '.com' after '@'
feature5: TODO: no '@' or more than 1 '@' in string
feature6: TODO: strings before and after '@' contain no alphabet and digit
feature7: TODO: '.' in consecutive after '@'
feature8: TODO: no string before and after '.' or '.' is near '@'
feature9: TODO: string contains 'not'
feature10: TODO: string has uppercase alphabet

Accuracy of your model: TODO: 0.9615384615384616
"""

import numpy as np


WEIGHT = [                           # The weight vector selected by you
	[0.2],                           # (Please fill in your own weights)
	[0.1],
	[0.1],
	[0.1],
	[-1],
	[-1],
	[-1],
	[-1],
	[-0.8],
	[-0.6]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	"""
	:return: the accuracy of the program distinguishing valid and invalid email address
	"""
	maybe_email_list = read_in_data()
	score = []
	for maybe_email in maybe_email_list:
		weight_vector = np.array(WEIGHT)  # np.array is faster because it uses parallel computing
		feature_vector = np.array(feature_extractor(maybe_email))
		score.append(weight_vector.T.dot(feature_vector))
	print(f'Accuracy: {accuracy(score)}')


def accuracy(score):
	"""
	:param score: list, containing the score of each email address
	:return: float, the accuracy of the program judging the validity of email address
	"""
	total = 0
	for i in range(len(score)):
		if i < len(score)//2:
			if score[i][0] <= 0:
				total += 1
		else:
			if score[i][0] > 0:
				total += 1
	return total/len(score)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with value 0's and 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if len(maybe_email.split('@')) == 2 else 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) != 0 else 0
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) != 0 else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.com' in maybe_email.split('@')[1] else 0
		elif i == 4:
			feature_vector[i] = 1 if not feature_vector[0] else 0
		elif i == 5:
			if feature_vector[0]:
				correct = 0
				for ch in maybe_email.split('@')[0]:
					if ch.isalpha() or ch.isdigit():
						correct += 1
						break
				for ch in maybe_email.split('@')[1]:
					if ch.isalpha() or ch.isdigit():
						correct += 1
						break
				feature_vector[i] = 1 if correct != 2 else 0
		elif i == 6:
			if feature_vector[0]:
				feature_vector[i] = 1 if '..' in maybe_email.split('@')[1] else 0
		elif i == 7:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0].split('.')[0]) == 0 or \
					len(maybe_email.split('@')[0].split('.')[-1]) == 0 or \
					len(maybe_email.split('@')[1].split('.')[0]) == 0 or \
					len(maybe_email.split('@')[1].split('.')[-1]) == 0 else 0
		elif i == 8:
			feature_vector[i] = 1 if 'not' in maybe_email else 0
		elif i == 9:
			no_upper = True
			for ch in maybe_email:
				if ch.isupper():
					no_upper = False
					break
			feature_vector[i] = 1 if not no_upper else 0
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that may be valid email addresses
	"""
	email_list = []
	with open(DATA_FILE, 'r') as f:
		for line in f:
			email_list.append(line.strip())
	return email_list


if __name__ == '__main__':
	main()
