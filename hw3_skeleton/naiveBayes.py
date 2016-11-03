'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
'''

import numpy as np

class NaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
	self.useLaplaceSmoothing = useLaplaceSmoothing      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
	n,d = X.shape
	self.classes = np.unique(y)
	self.numClasses = len(self.classes)
	
	# K x d matrix for predicting probs
	self.conditional_probs = np.zeros([self.numClasses,d])
	self.probs = np.zeros(self.numClasses)
	#X[i,j] is the number of times feature j occurs in instance i
	#For each label yk
		#estimate P(Y=yk)
		#for each X[i,j] estimate P(Xi = Xi,j | Y = yk)

	for i, curClass in enumerate(self.classes):
		item = X[np.logical_or.reduce([y == curClass])]
		self.probs[i] = item.shape[0] / float(n)
	        if (self.useLaplaceSmoothing):
			self.conditional_probs[i, :] = (1.0 + np.sum(item, axis=0) ) / (d + np.sum(item))
		else:
			self.conditional_probs[i,:] = np.sum(item, axis=0) / np.sum(item)

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
	#argmax for yk of
		#log P(Y=yk) + sum 1->d log P(Xj = xj | Y = yk)
	probs = self.predictProbs(X)
	return self.classes[np.argmax(probs, axis=1)]   


    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
	#nxd dot a d x k
	probs = X.dot(np.log(self.conditional_probs).T)
	probs += np.log(self.probs) 
        probs -= np.mean(probs)
	probs = np.exp(probs)
	#normalize
	sums = probs.sum(axis=1)
	normalized = probs/sums[:,np.newaxis]
        return normalized
        
class OnlineNaiveBayes:

    def __init__(self, useLaplaceSmoothing=True):
        '''
        Constructor
        '''
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
    
    
    def predictProbs(self, X):
        '''
        Used the model to predict a vector of class probabilities for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-by-K numpy array of the predicted class probabilities (for K classes)
        '''
