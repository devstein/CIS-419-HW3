'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from sklearn import tree

class BoostedDT:
	
    def __init__(self, numBoostingIters=100, maxTreeDepth=3):
        '''
        Constructor
        '''
        #TODO
	self.numBoostingIters = numBoostingIters
	self.maxTreeDepth = maxTreeDepth
    	self.classifiers = np.empty([numBoostingIters], dtype = tree.DecisionTreeClassifier)
	self.betas = np.zeros(numBoostingIters)

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        #TODO
	n,d = X.shape
	self.classes = np.unique(y)
	self.numClasses = len(self.classes)
	#initialize vector of n uniform weights w1 -> 1/n
	w = np.full(n, 1.0/float(n))
	for t in range(self.numBoostingIters):
		#train model on X, y with weights wt
		self.classifiers[t] = tree.DecisionTreeClassifier(max_depth = self.maxTreeDepth)
		self.classifiers[t] = self.classifiers[t].fit(X,y, sample_weight=w)
		predicted = self.classifiers[t].predict(X)
		err = 0
		for i in range(n):
			err += (y[i] != predicted[i]) * w[i]
		#do Beta
		self.betas[t] = .5 * ( np.log( (1.0 - err)/float(err) ) + np.log(self.numClasses - 1) )
		#update all instances of weight
		for i in range(n):
			if (y[i] == predicted[i]): w[i] *= np.exp(-1 * self.betas[t])
			else: w[i] *= np.exp(self.betas[t])
		#normalize w_{t+1}
                w = w / np.sum(w)

    def predict(self, X):
        '''
        Used the modeld to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        #TODO
	n,d = X.shape
	values = np.zeros((n))
	betas = np.zeros((n,self.numClasses))
	#loop over all classifiers
	#for each classifier add it to each sum if prediction is correct 
	for t in range(self.numBoostingIters):
		prediction = self.classifiers[t].predict(X)
		for i in range(self.numClasses):
			betas[:,i] += (prediction == self.classes[i])* self.betas[t]
	return self.classes[np.argmax(betas, axis=1)]
