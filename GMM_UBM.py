import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

#import model1
#from model1 import models1

import plot2
from plot2 import plot_curve

import glob

import _pickle
from scipy.io.wavfile import read
#from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.externals import joblib

from tsfresh.examples import load_robot_execution_failures
from tsfresh import extract_features
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#fig = plt.figure()
#ax = plt.axes(projection='3d')

 
import librosa
import librosa.display
import IPython
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import padasip as pa 
from scipy.stats import multivariate_normal as mvn


from mpl_toolkits import mplot3d

from scipy.stats import multivariate_normal as mvn
import math

from numpy.linalg import slogdet, det, solve
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
import time

from sklearn.datasets import load_digits
from sklearn.mixture.base import BaseMixture
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import os, sys, email,re
from nltk.corpus import stopwords

from matplotlib.patches import Ellipse


def calculate_delta(array):

	rows,cols = array.shape
	deltas = np.zeros((rows,cols))
	N = 2
	for i in range(rows):
		index = []
		j = 1
		while j <= N:
			if i-j < 0:
				first = 0
			else:
				first = i-j
			if i+j > rows -1:
				second = rows -1
			else:
				second = i+j
			index.append((second,first))
			j+=1
		deltas[i] = ( array[index[0][0]]-array[index[0][1]] + (2 * (array[index[1][0]]-array[index[1][1]])) ) / 10
	return deltas





def map_adaptation(gmm, data, max_iterations = 10, likelihood_threshold = 1e-20, relevance_factor = 16):
	N = data.shape[0]
	D = data.shape[1]
	K = gmm.C
    
	mu_new = np.zeros((K,D))
	n_k = np.zeros((K,1))
    
	mu_k = gmm.mu
	cov_k = gmm.sigma
	pi_k = gmm.pi

	old_likelihood = gmm.score(data)
	new_likelihood = 0
	iterations = 0
	while((abs(old_likelihood - new_likelihood)).all() > likelihood_threshold and iterations < max_iterations):
		iterations += 1
		old_likelihood = new_likelihood
		z_n_k = gmm.predict_proba(data)
		n_k = np.sum(z_n_k,axis = 0)

		for i in range(K):
			temp = np.zeros((1,D))
			for n in range(N):
				temp += z_n_k[n][i]*data[n,:]
			mu_new[i] = (1/n_k[i])*temp

		adaptation_coefficient = n_k/(n_k + relevance_factor)
		for k in range(K):
			mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
		gmm.mu = mu_k

		log_likelihood = gmm.score(data)
		new_likelihood = log_likelihood
		#print(log_likelihood)
	#print(gmm.mu)
	return gmm


def unit_gaussian(x,mu,sigma):
	inv_cov = np.linalg.inv(sigma)
	D = mu.shape[0]
	exponent = np.exp((-0.5)*np.dot(np.dot((x - mu),inv_cov),(x - mu).T))
	z = 1/(((2*np.pi)**(D/2))*(np.linalg.det(sigma)**0.5))
	return z*exponent


class GMM:
	
	def __init__(self, C, n_runs):
		self.C = C # number of Guassians/clusters
		self.n_runs = n_runs
	
	def get_params(self):
		return (self.mu, self.pi, self.sigma)


	def score(self,data):
		log_likelihood = 0
		N = len(data)
		for n in range(N):
			temp = 0
			for k in range(self.C):
				temp += self.pi[k] * (unit_gaussian(data[n],self.mu[k,:],self.sigma[k,:,:]))
			log_likelihood += np.log(temp)
		return log_likelihood	



	def calculate_mean_covariance(self, X, prediction):
		d = X.shape[1]
		labels = np.unique(prediction)
		self.initial_means = np.zeros((self.C, d))
		self.initial_cov = np.zeros((self.C, d, d))
		self.initial_pi = np.zeros(self.C)
        
		counter=0
		for label in labels:
			ids = np.where(prediction == label) #returns indices
			self.initial_pi[counter] = len(ids[0]) / X.shape[0]
			self.initial_means[counter,:] = np.mean(X[ids], axis = 0)
			de_meaned = X[ids] - self.initial_means[counter,:]
			Nk = X[ids].shape[0] # number of data points in current gaussian
			self.initial_cov[counter,:, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
			counter+=1

		print((self.initial_pi).sum())            
		#assert np.sum(self.initial_pi) == 1.0    
		return (self.initial_means, self.initial_cov, self.initial_pi)
    
    
    
	def _initialise_parameters(self, X):
		n_clusters = self.C
		kmeans = KMeans(n_clusters= n_clusters, init="k-means++", max_iter=500, algorithm = 'auto')
		fitted = kmeans.fit(X)
		prediction = kmeans.predict(X)

		self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)
        
		return (self._initial_means, self._initial_cov, self._initial_pi)
            
        
        
	def _e_step(self, X, pi, mu, sigma):

		N = X.shape[0] 
		self.gamma = np.zeros((N, self.C))

		const_c = np.zeros(self.C)
        
        
		self.mu = self.mu if self._initial_means is None else self._initial_means
		self.pi = self.pi if self._initial_pi is None else self._initial_pi
		self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

		for c in range(self.C):
			self.gamma[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])

		gamma_norm = np.sum(self.gamma, axis=1)[:,np.newaxis]
		self.gamma /= gamma_norm

		return self.gamma
    
    
	def _m_step(self, X, gamma):

		N = X.shape[0] # number of objects
		C = self.gamma.shape[1] # number of clusters
		d = X.shape[1] # dimension of each object

		self.pi = np.mean(self.gamma, axis = 0)

		self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis = 0)[:,np.newaxis]

		for c in range(C):
			x = X - self.mu[c, :] # (N x d)
            
			gamma_diag = np.diag(self.gamma[:,c])
			x_mu = np.matrix(x)
			gamma_diag = np.matrix(gamma_diag)

			sigma_c = x.T * gamma_diag * x
			self.sigma[c,:,:]=(sigma_c) / np.sum(self.gamma, axis = 0)[:,np.newaxis][c]

		return self.pi, self.mu, self.sigma
    
    
	def _compute_loss_function(self, X, pi, mu, sigma):

		N = X.shape[0]
		C = self.gamma.shape[1]
		self.loss = np.zeros((N, C))

		for c in range(C):
			dist = mvn(self.mu[c], self.sigma[c],allow_singular=True)
			self.loss[:,c] = self.gamma[:,c] * (np.log(self.pi[c]+0.00001)+dist.logpdf(X)-np.log(self.gamma[:,c]+0.000001))
		self.loss = np.sum(self.loss)
		return self.loss
    
    
    
	def fit(self, X):
        
		d = X.shape[1]
		self.mu, self.sigma, self.pi =  self._initialise_parameters(X)
        
		try:
			for run in range(self.n_runs):  
				self.gamma  = self._e_step(X, self.mu, self.pi, self.sigma)
				self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
				loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)
                
				#if run % 10 == 0:
					#print("Iteration: %d Loss: %0.6f" %(run, loss))
        
		except Exception as e:
			print(e)
                    
		return self
    
   
	def predict(self, X):
		labels = np.zeros((X.shape[0], self.C))
        
		for c in range(self.C):
			labels [:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])
		labels  = labels.argmax(1)
		return labels 
    
	def predict_proba(self, X):

		post_proba = np.zeros((X.shape[0], self.C))
        
		for c in range(self.C):
			post_proba[:,c] = self.pi[c] * mvn.pdf(X, self.mu[c,:], self.sigma[c])
		
		return post_proba



features = np.asarray(())

#modelpath = '/home/sahaja/Desktop/person/'

path = '/home/sahaja/Desktop/person/*.wav'   
files = glob.glob(path)   
for file in files: 
	#print(file)
	sr,audio = read(file)
	mfcc_feat = mfcc.mfcc(audio,16000, 0.025, 0.01,20,appendEnergy = True)    
	mfcc_feat = preprocessing.scale(mfcc_feat)
	delta = calculate_delta(mfcc_feat)
	vector = np.hstack((mfcc_feat,delta))
			
	if features.size == 0:
		features = vector
	else:
		features = np.vstack((features, vector))


sklearn_pca = PCA(n_components = 39)
sklearn_pca.fit(features)
Y_sklearn = sklearn_pca.fit_transform(features)

#ax.scatter3D(Y_sklearn[:,0], Y_sklearn[:,1] ,Y_sklearn[:,2], c=Y_sklearn[:,2], cmap='Greens');
#plt.show()



model = GMM(3,30)


fitted_values = model.fit(Y_sklearn)
#print(fitted_values)
time.sleep(1)
predicted_values = model.predict(Y_sklearn)
time.sleep(1)
#post = model.predict_proba(Y_sklearn)
#print((model.mu).shape)
#print(model.mu)

#plot_curve(model, Y_sklearn, predicted_values)



#loading the trained model

filename = 'modelling.pkl'
#joblib.dump(model,filename)
gaussian = joblib.load(filename)



#speaker models

modls = [0]*2
features1 = np.asarray(())


#input files 

path = '/home/sahaja/Desktop/person/speaker_models/train1/*.wav'   
files = glob.glob(path)   
for file in files: 
	sr,audio = read(file)
	mfcc_feat = mfcc.mfcc(audio,16000, 0.025, 0.01,20,appendEnergy = True)    
	mfcc_feat = preprocessing.scale(mfcc_feat)
	delta = calculate_delta(mfcc_feat)
	vector = np.hstack((mfcc_feat,delta))
	print(vector)		
	if features1.size == 0:
		features1 = vector
	else:
		features1 = np.vstack((features1, vector))


#print(vector)

#print(features1)
#print(features1.shape)
sklearn_pca = PCA(n_components = 39)
sklearn_pca.fit(features1)
Y_sklearn1 = sklearn_pca.fit_transform(features1)



modls[0] = map_adaptation(gaussian, Y_sklearn1, max_iterations = 10, likelihood_threshold = 1e-20, relevance_factor = 16)
#print(modls[0].mu)
fitted_values1 = modls[0].fit(Y_sklearn1)
predicted_values1 = modls[0].predict(Y_sklearn1)
post1 = modls[0].predict_proba(Y_sklearn1)

#filename0 = 'model0.pkl'
#joblib.dump(modls[0],filename0)



#plot_curve(modls[0],Y_sklearn,predicted_values1)



features2 = np.asarray(())

path = '/home/sahaja/Desktop/person/speaker_models/train2/*.wav'   
files=glob.glob(path)   
for file in files: 
	sr,audio = read(file)
	mfcc_feat = mfcc.mfcc(audio,16000, 0.025, 0.01,20,appendEnergy = True)    
	mfcc_feat = preprocessing.scale(mfcc_feat)
	delta = calculate_delta(mfcc_feat)
	vector = np.hstack((mfcc_feat,delta))
			
	if features2.size == 0:
		features2 = vector
	else:
		features2 = np.vstack((features2, vector))

sklearn_pca = PCA(n_components = 39)
sklearn_pca.fit(features2)
Y_sklearn2 = sklearn_pca.fit_transform(features2)



modls[1] = map_adaptation(model, Y_sklearn2, max_iterations = 10, likelihood_threshold = 1e-20, relevance_factor = 16)
#print(modls[1].mu)
fitted_values2 = modls[1].fit(Y_sklearn2)
predicted_values2 = modls[1].predict(Y_sklearn2)
post2 = modls[1].predict_proba(Y_sklearn2)


for j in range(0,2):

	gmm    = modls[j]         #checking with each model one by one
	#print(modls[i].mu)
	scores = np.array(gmm.score(Y))
	#print(gmm.mu)
	#print(scores)
	log_likelihood[j] = scores.sum()

"""
#filename1 = 'model1.pkl'
#joblib.dump(modls[1],filename1)

"""



 
# some time later...
 
# load the model from disk
#loaded_model = joblib.load(filename)
#result = loaded_model.score(X_test, Y_test)
#print(result)


#picklefile = file.split("-")[0]+".gmm"
#_pickle.dump(model,open(picklefile,'w'))
#print ('+ modeling completed for speaker:',picklefile," with data point = ",features.shape )


#gmm_files = [os.path.join(modelpath,fname) for fname in 
#             os.listdir(modelpath) if fname.endswith('.gmm')]

#Load the Gaussian gender Models
#models    = [cPickle.load(open(fname,'r')) for fname in gmm_files]
#speakers   = [fname.split("")[-1].split(".gmm")[0] for fname 
#in gmm_files]

#plot_curve(model,Y_sklearn,predicted_values)
















