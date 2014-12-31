from collections import Counter
from random import *
from math import log

with open('hw12Data/emailDataset/testFeatures.csv') as f:
	inp = f.readlines()
testFeatures = []
for line in inp:
	testFeatures.append([float(x) for x in line.split(',')])

with open('hw12Data/emailDataset/valFeatures.csv') as f:
	inp = f.readlines()
valFeatures = []
for line in inp:
	valFeatures.append([float(x) for x in line.split(',')])

with open('hw12Data/emailDataset/valLabels.csv') as f:
	inp = f.readlines()
valLabels = []
for line in inp:
	valLabels.append(int(line))

valData = []
for i in range(len(valFeatures)):
	valData.append((valFeatures[i], valLabels[i]))

with open('hw12Data/emailDataset/trainFeatures.csv') as f:
	inp = f.readlines()
trainFeatures = []
for line in inp:
	trainFeatures.append([float(x) for x in line.split(',')])

with open('hw12Data/emailDataset/trainLabels.csv') as f:
	inp = f.readlines()
trainLabels = []
for line in inp:
	trainLabels.append(int(line))

trainingData = []
for i in range(len(trainFeatures)):
	trainingData.append((trainFeatures[i], trainLabels[i]))

class Node:
	def __init__(self, deci, coord, threshold, left, right):
		self.decision = deci
		self.coordinate = coord
		self.threshold = threshold
		self.left = left
		self.right = right
	def is_leaf(self):
		return self.decision != None

def entropy(data):
	zeroCount = 0.0
	oneCount = 0.0
	for item in data:
		if item[1] == 0:
			zeroCount += 1
		else:
			oneCount += 1
	zeroProb = zeroCount/len(data)
	oneProb = oneCount/len(data)
	if zeroProb == 0 or oneProb == 0:
		return 0
	return -(zeroProb*log(zeroProb, 2) + oneProb*log(oneProb, 2))

def build_tree(trainingSet, depth):
	same_labels = True
	ref = trainingSet[0][1]
	for item in trainingSet:
		if ref != item[1]:
			same_labels = False
	if same_labels:
		return Node(trainingSet[0][1], None, None, None, None)
	if depth > 15:
		zeroCount = 0
		for item in trainingSet:
			if item[1] == 0:
				zeroCount += 1
		if zeroCount > len(trainingSet)/2:
			return Node(0, None, None, None, None)
		else:
			return Node(1, None, None, None, None)
	bestFeature = None
	bestCutoff = None
	lowestEntropy = float('inf')
	bestIndex = None
	for i in sample(range(len(trainingSet[0][0])), 8):
		trainingSet.sort(key=lambda x: x[0][i])
		for j in range(len(trainingSet) - 1):
			cutoff = (trainingSet[j][0][i] + trainingSet[j+1][0][i]) / 2
			leftHalf = trainingSet[:j+1]
			rightHalf = trainingSet[j+1:]
			ent = entropy(leftHalf)*(float(len(leftHalf))/len(trainingSet)) + \
			entropy(rightHalf)*(float(len(rightHalf))/len(trainingSet))
			if ent < lowestEntropy:
				lowestEntropy = ent
				bestFeature = i
				bestCutoff = cutoff
				bestIndex = j+1
	trainingSet.sort(key=lambda x: x[0][bestFeature])
	leftTree = build_tree(trainingSet[:bestIndex], depth + 1)
	rightTree = build_tree(trainingSet[bestIndex:], depth + 1)
	return Node(None, bestFeature, bestCutoff, leftTree, rightTree)

def traverse(features, tree):
	while not tree.is_leaf():
		if features[tree.coordinate] <= tree.threshold:
			tree = tree.left
		else:
			tree = tree.right
	return tree.decision

def sample_with_replacement(testFeatures, num):
	sampledInd = set()
	for i in range(num):
		sampledInd.add(randint(0, len(testFeatures) - 1))
	sampled = []
	for ind in sampledInd:
		sampled.append(testFeatures[ind])
	return sampled

for i in [1, 2, 5, 10, 25]:
	with open('emailOutput{}.csv'.format(i), 'w') as f:
		results = []
		trees = []
		for j in range(i):
			trees.append(build_tree(
				sample_with_replacement(trainingData, len(trainingData)), 0))
		for features in valFeatures:
			classifications = []
			for tree in trees:
				classifications.append(traverse(features, tree))
			counts = Counter([0, 1])
			for c in classifications:
				counts[c] += 1
			classification = counts.most_common(1)[0][0]
			f.write(str(classification) + '\n')
			results.append(classification)
	numCorrect = 0.0
	for k in range(len(results)):
		if results[k] == valLabels[k]:
			numCorrect += 1
	print(numCorrect/len(results))
