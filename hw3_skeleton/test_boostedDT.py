"""
======================================================
Test the boostedDT against the standard decision tree
======================================================

Author: Eric Eaton, 2014

"""
print(__doc__)

import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier

from boostedDT import BoostedDT

# load the data set
iris = datasets.load_iris()
X = iris.data
y = iris.target

n,d = X.shape
nTrain = 0.5*n  #training on 50% of the data

# shuffle the data
idx = np.arange(n)
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# split the data
Xtrain = X[:nTrain,:]
ytrain = y[:nTrain]
Xtest = X[nTrain:,:]
ytest = y[nTrain:]

# train the decision tree
modelDT = DecisionTreeClassifier()
modelDT.fit(Xtrain,ytrain)

# train the boosted DT
modelBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
modelBoostedDT.fit(Xtrain,ytrain)

# output predictions on the remaining data
ypred_DT = modelDT.predict(Xtest)
ypred_BoostedDT = modelBoostedDT.predict(Xtest)
# compute the training accuracy of the model
accuracyDT = accuracy_score(ytest, ypred_DT)
accuracyBoostedDT = accuracy_score(ytest, ypred_BoostedDT)

print "Decision Tree Accuracy = "+str(accuracyDT)
print "Boosted Decision Tree Accuracy = "+str(accuracyBoostedDT)

# challenge data
challengeTrainingData = np.loadtxt('data/challengeTrainLabeled.dat', delimiter=',')
n,d = challengeTrainingData.shape
challengeTrainX = challengeTrainingData[:,:d-1]
challengeTrainY = challengeTrainingData[:,d-1]

challengeTestData = np.loadtxt('data/challengeTestUnlabeled.dat', delimiter=',')
n,d = challengeTestData.shape
challengeTestX = challengeTestData[:,:]


challengeBoostedDT = BoostedDT(numBoostingIters=100, maxTreeDepth=2)
challengeBoostedDT.fit(challengeTrainX, challengeTrainY)

ypred_challenge = challengeBoostedDT.predict(challengeTestX)
output = ""
for i, x in enumerate(ypred_challenge):
	output += str(x)
	if (i != len(ypred_challenge) - 1): output += ','

svm = LinearSVC() 
svm.fit(challengeTrainX, challengeTrainY)
ypred_svm = svm.predict(challengeTrainX)
print "SVM: ", str(accuracy_score(challengeTrainY, ypred_svm))

knear = KNeighborsClassifier()
knear.fit(challengeTrainX, challengeTrainY)
ypred_knear = knear.predict(challengeTrainX)
print "knear: ", str(accuracy_score(challengeTrainY, ypred_knear))

svc = SVC()
svc.fit(challengeTrainX, challengeTrainY)
ypred_svc = svc.predict(challengeTrainX)
print "SVC: ", str(accuracy_score(challengeTrainY, ypred_svc))

adaboost = AdaBoostClassifier()
adaboost.fit(challengeTrainX, challengeTrainY)
ypred_ada = adaboost.predict(challengeTrainX)
print "ADA: ", str(accuracy_score(challengeTrainY, ypred_ada))


