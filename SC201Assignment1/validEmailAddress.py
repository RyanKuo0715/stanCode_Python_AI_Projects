"""
File: validEmailAddress.py
Name: Ryan Kuo
----------------------------
This file shows what a feature vector is
and what a weight vector is for valid email 
address classifier. You will use a given 
weight vector to classify what is the percentage
of correct classification.

Accuracy of this model: TODO: 0.6538461538461539
"""

WEIGHT = [                           # The weight vector selected by Jerry
	[0.4],                           # (see assignment handout for more details)
	[0.4],
	[0.2],
	[0.2],
	[0.9],
	[-0.65],
	[0.1],
	[0.1],
	[0.1],
	[-0.7]
]

DATA_FILE = 'is_valid_email.txt'     # This is the file name to be processed


def main():
	"""
	:return: the accuracy of the program distinguishing valid and invalid email address
	"""
	maybe_email_list = read_in_data()
	score = []
	for maybe_email in maybe_email_list:
		feature_vector = feature_extractor(maybe_email)
		score.append(sum(WEIGHT[i][0]*feature_vector[i] for i in range(len(WEIGHT))))
	print(f'Accuracy: {accuracy(score)}')


def accuracy(score):
	"""
	:param score: list, containing the score of each email address
	:return: float, the accuracy of the program judging the validity of email address
	"""
	total = 0
	for i in range(len(score)):
		if i < len(score)//2:
			if score[i] <= 0:
				total += 1
		else:
			if score[i] > 0:
				total += 1
	return total/len(score)


def feature_extractor(maybe_email):
	"""
	:param maybe_email: str, the string to be processed
	:return: list, feature vector with 10 values of 0's or 1's
	"""
	feature_vector = [0] * len(WEIGHT)
	for i in range(len(feature_vector)):
		if i == 0:
			feature_vector[i] = 1 if '@' in maybe_email else 0
		elif i == 1:
			if feature_vector[0]:
				feature_vector[i] = 1 if '.' not in maybe_email.split('@')[0] else 0
		elif i == 2:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[0]) != 0 else 0
		elif i == 3:
			if feature_vector[0]:
				feature_vector[i] = 1 if len(maybe_email.split('@')[1]) != 0 else 0
		elif i == 4:
			if feature_vector[0]:
				for j in range(1, len(maybe_email.split('@'))):
					if '.' in maybe_email.split('@')[j]:
						feature_vector[i] = 1
		elif i == 5:
			feature_vector[i] = 1 if ' ' not in maybe_email else 0
		elif i == 6:
			if '.com' in maybe_email:
				feature_vector[i] = 1 if len(maybe_email.split('.com')[1]) == 0 else 0
		elif i == 7:
			if '.edu' in maybe_email:
				feature_vector[i] = 1 if len(maybe_email.split('.edu')[1]) == 0 else 0
		elif i == 8:
			if '.tw' in maybe_email:
				feature_vector[i] = 1 if len(maybe_email.split('.tw')[1]) == 0 else 0
		elif i == 9:
			feature_vector[i] = 1 if len(maybe_email) > 10 else 0
	return feature_vector


def read_in_data():
	"""
	:return: list, containing strings that might be valid email addresses
	"""
	email_list = []
	with open(DATA_FILE, 'r') as f:
		for line in f:
			email_list.append(line.strip())
	return email_list


if __name__ == '__main__':
	main()
