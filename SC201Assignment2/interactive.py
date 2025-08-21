"""
File: interactive.py
Name: Ryan Kuo
------------------------
This file uses the function interactivePrompt
from util.py to predict the reviews input by 
users on Console. Remember to read the weights
and build a Dict[str: float]
"""

from submission import *


def main():
	featureExtractor = extractWordFeatures
	weights = {}
	with open('weights', 'r', encoding='utf-8') as f:
		# encoding='utf-8' to solve UnicodeDecodeError
		# https://oxygentw.net/blog/computer/python-file-utf8-encoding/
		for line in f:
			line = line.strip()
			weights[line.split()[0]] = float(line.split()[1])
	interactivePrompt(featureExtractor, weights)
	# test: although this movie suffers from some cliche , this film is still worth watching


if __name__ == '__main__':
	main()
