import numpy as np
from sklearn import preprocessing
import python_speech_features as mfcc

#import model1

import _pickle
import numpy as np
from scipy.io.wavfile import read
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

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
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt

#from matplotlib import cm
#from colorspacious import cspace_converter
#from collections import OrderedDict

#cmaps = OrderedDict()

import seaborn as sns
import padasip as pa 
from scipy.stats import multivariate_normal as mvn


from mpl_toolkits import mplot3d



import numpy as np 
import pandas as pd 
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

def plot_curve(model,Y_sklearn,predicted_values):

	centers = np.zeros((3,39))
	for i in range(model.C): #model.C
		density = mvn(cov=model.sigma[i], mean=model.mu[i]).logpdf(Y_sklearn)
		centers[i, :] = Y_sklearn[np.argmax(density)]
    
	plt.figure(figsize = (12,8))

	plt.scatter(Y_sklearn[:, 0], Y_sklearn[:, 1],c=predicted_values ,s=20, cmap='viridis', zorder=1)

	plt.scatter(centers[:, 0], centers[:, 1],c='black', s=300, alpha=0.5, zorder=2);

	plt.show()