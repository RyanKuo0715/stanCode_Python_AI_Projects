#!/usr/bin/python

import math
import random
from collections import defaultdict
from util import *
from typing import Any, Dict, Tuple, List, Callable

FeatureVector = Dict[str, int]
WeightVector = Dict[str, float]
Example = Tuple[FeatureVector, int]


############################################################
# Milestone 3a: feature extraction

def extractWordFeatures(x: str) -> FeatureVector:
    """
    Extract word features for a string x. Words are delimited by
    whitespace characters only.
    @param string x: a string whose number of each different word will be counted
    @return dict: feature vector representation of x.
    Example: "I am what I am" --> {'I': 2, 'am': 2, 'what': 1}
    """
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    feature_vector = defaultdict(int)  # int(), float(), list()
    for word in x.split():
        feature_vector[word] += 1
    return feature_vector
    # END_YOUR_CODE


############################################################
# Milestone 4: Sentiment Classification

def learnPredictor(trainExamples: List[Tuple[Any, int]], validationExamples: List[Tuple[Any, int]],
                   featureExtractor: Callable[[str], FeatureVector], numEpochs: int, alpha: float) -> WeightVector:
    """
    Given |trainExamples| and |validationExamples| (each one is a list of (x,y)
    pairs), a |featureExtractor| to apply to x, and the number of epochs to
    train |numEpochs|, the step size |eta|, return the weight vector (sparse
    feature vector) learned.

    You should implement gradient descent.
    Note: only use the trainExamples for training!
    You should call evaluatePredictor() on both trainExamples and validationExamples
    to see how you're doing as you learn after each epoch. Note also that the 
    identity function may be used as the featureExtractor function during testing.
    """
    weights = {}  # the weight vector

    # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
    def predictor(movie_review: str) -> int:
        return 1 if (dotProduct(weights, featureExtractor(movie_review))) > 0 else -1
    for epoch in range(numEpochs):
        for x, y in trainExamples:
            feature_vector = featureExtractor(x)
            y = 1 if y == 1 else 0
            k = dotProduct(weights, feature_vector)
            h = 1 / (1 + math.exp(-k))
            # # Ryan's Method
            # gradient = {key: (h - y) * value for key, value in feature_vector.items()}
            # increment(weights, -alpha, gradient)
            # TA's Method
            # w_j = w_j - alpha*(h-y)*x_j
            # d1[key] = d1.get(key, 0) + scale*d2[key]
            increment(weights, -alpha*(h-y), feature_vector)
        # print(f'Training Error: ({epoch} epoch): '
        #       f'{evaluatePredictor(trainExamples, lambda movie_review:1 if dotProduct(weights, featureExtractor(movie_review)) >= 0 else -1)}')
        print(f'Training Error: ({epoch} epoch): {evaluatePredictor(trainExamples, predictor)}')
        print(f'Validation Error: ({epoch} epoch): {evaluatePredictor(validationExamples, predictor)}')
    # END_YOUR_CODE
    return weights


############################################################
# Milestone 5a: generate test case

def generateDataset(numExamples: int, weights: WeightVector) -> List[Example]:
    """
    Return a set of examples (phi(x), y) randomly which are classified correctly by
    |weights|.
    """
    random.seed(42)

    def generateExample() -> Tuple[Dict[str, int], int]:
        """
        Return a single example (phi(x), y).
        phi(x) should be a dict whose keys are a subset of the keys in weights
        and values are their word occurrence.
        y should be 1 or -1 as classified by the weight vector.
        Note that the weight vector can be arbitrary during testing.
        """
        # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
        phi = defaultdict(int)
        # for _ in range(random.randint(1, len(weights))):  # Python這樣寫沒問題，但其他程式會有動態迴圈的問題，每次迴圈跑的次數會不一樣
        #     phi[random.choice(list(weights.keys()))] += 1
        len_random_weight = random.randint(1, len(weights))
        for _ in range(len_random_weight):
            phi[random.choice(list(weights.keys()))] += 1
        y = 1 if dotProduct(phi, weights) > 0 else -1
        # END_YOUR_CODE
        return phi, y

    return [generateExample() for _ in range(numExamples)]


############################################################
# Milestone 5b: character features

def extractCharacterFeatures(n: int) -> Callable[[str], FeatureVector]:
    """
    Return a function that takes a string |x| and returns a sparse feature
    vector consisting of all n-grams of |x| without spaces mapped to their n-gram counts.
    EXAMPLE: (n = 3) "I like tacos" --> {'Ili': 1, 'lik': 1, 'ike': 1, ...
    You may assume that n >= 1.
    """

    def extract(x: str) -> Dict[str, int]:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        feature_character_vector = defaultdict(int)
        x_new = x.replace(' ', '')
        # x-new = ''.join(x.split())
        for i in range(len(x_new)-n+1):
            feature_character_vector[x_new[i:i+n]] += 1
        return feature_character_vector
        # END_YOUR_CODE

    return extract


############################################################
# Problem 3f: 
def testValuesOfN(n: int):
    """
    Use this code to test different values of n for extractCharacterFeatures
    This code is exclusively for testing.
    Your full written solution for this problem must be in sentiment.pdf.
    """
    trainExamples = readExamples('polarity.train')
    validationExamples = readExamples('polarity.dev')
    featureExtractor = extractCharacterFeatures(n)
    weights = learnPredictor(trainExamples, validationExamples, featureExtractor, numEpochs=20, alpha=0.01)
    outputWeights(weights, 'weights')
    outputErrorAnalysis(validationExamples, featureExtractor, weights, 'error-analysis')  # Use this to debug
    trainError = evaluatePredictor(trainExamples,
                                   lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    validationError = evaluatePredictor(validationExamples,
                                        lambda x: (1 if dotProduct(featureExtractor(x), weights) >= 0 else -1))
    print(("Official: train error = %s, validation error = %s" % (trainError, validationError)))
