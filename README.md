# Automatic Speech Recognition

- The data used is taken from VoxForge randomly .

- The data consist of audio files of 10-20 sec for each speaker. Pre-processed the data On Mel scale using filter banks and extracted MFCC as features

- This is a python approach towards achieving speaker verification using GMM on UBM . A text-independent Speaker Verification Model by building an UBM using GMM which was converged by using Expectation Maximization(EM) algorithm on entire dataset .

- Using the UBM and separate dataset for each speaker, the individual speaker models are created (here 2 speaker models are used) and adapted speaker models using Maximum-a-posteriori(MAP) adaptation.

- Then log likelihood is used over a random threshold to calculate the final result to identify the whether the right person is speaking or not . Alongside this a code is written to plot the required curves .



