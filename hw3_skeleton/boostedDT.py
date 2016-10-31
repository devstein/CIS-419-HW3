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
	self.numClasses = len(np.unique(y))
	#initialize vector of n uniform weights w1 -> 1/n
	w = np.full(n, 1.0/float(n))
	for t in range(self.numBoostingIters):
		#train model on X, y with weights wt
		self.classifiers[t] = tree.DecisionTreeClassifier(max_depth = self.maxTreeDepth)
		model = self.classifiers[t].fit(X,y, sample_weight=w)
		predicted = model.predict(X)
		print predicted
		err = 0
		for i in range(n):
			if (y[i] != predicted[i]):
				err += w[i]
		#do Beta
		#beta = .5 * ( np.log( (1 - err)/float(err) ) + np.log(self.numClasses - 1) )
		beta = np.log( (1 - err)/float(err) ) + np.log(self.numClasses - 1)
		self.betas[t] = beta
		#print "Error: ", err, " Beta: ", self.betas[t]
		#update all instances of weight
		for i in range(n):
			#w[i] = w[i] * np.exp(-beta * y[i] * predicted[i] )
			#if (y[i] == predicted[i]): w[i] = w[i] * np.exp(-beta)
			#else: w[i] = w[i] * np.exp(beta)
			if (y[i] != predicted[i]): w[i] = w[i] * np.exp(beta)
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
	values = np.zeros(n)
	betas = np.zeros(self.numClasses)
	#loop over all classifiers
	#have numClasses sumchrome-extension://nkoccljplnhpfnfiajclkommnmllphnl/html/crosh.htmlis
	#for each classifier add it to each sum if prediction is correct
	print "BETAS: ", self.betas
	for i in range(n):
		for t in range(self.numBoostingIters):
			prediction = self.classifiers[t].predict(X[i])
			betas[prediction] += self.betas[t]
		#print "Betas: ", betas
		#print "Argmax: ", np.argmax(betas)
		values[i] = np.argmax(betas)
	print "predicting... ", values
	return values


