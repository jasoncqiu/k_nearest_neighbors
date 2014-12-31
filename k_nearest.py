from collections import Counter
from numpy import *

with open('hw12Data/digitsDataset/trainFeatures.csv') as f:
	inp = f.readlines()
trainFeatures = []
for line in inp:
	trainFeatures.append(array([float(x) for x in line.split(',')]))

with open('hw12Data/digitsDataset/trainLabels.csv') as f:
	inp = f.readlines()
trainLabels = []
for line in inp:
	trainLabels.append(float(line))

with open('hw12Data/digitsDataset/valFeatures.csv') as f:
	inp = f.readlines()
valFeatures = []
for line in inp:
	valFeatures.append(array([float(x) for x in line.split(',')]))

with open('hw12Data/digitsDataset/valLabels.csv') as f:
	inp = f.readlines()
valLabels = []
for line in inp:
	valLabels.append(float(line))
	
with open('hw12Data/digitsDataset/testFeatures.csv') as f:
	inp = f.readlines()
testFeatures = []
for line in inp:
	testFeatures.append(array([float(x) for x in line.split(',')]))

k = 1
with open('digitsOutput.csv', 'w') as f:
	classifications = []
	for i in range(len(testFeatures)):
		distances = []
		for j in range(len(trainFeatures)):
			arr = testFeatures[i] - trainFeatures[j]
			arr = arr*arr
			dist = sum(arr)
			distances.append((dist, trainLabels[j]))
		distances.sort(key=lambda x: x[0])
		nearest = []
		for ind in range(k):
			nearest.append(distances[ind][1])
		counts = Counter([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
		for label in nearest:
			counts[label] += 1
		classification = counts.most_common(1)[0][0]
		f.write(str(classification) + '\n')
		classifications.append(classification)

# total = len(valFeatures)
# numCorrect = 0.0
# for i in range(len(valLabels)):
# 	if classifications[i] == valLabels[i]:
# 		numCorrect += 1
# print(numCorrect/total)
