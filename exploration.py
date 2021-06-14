#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 09:43:29 2021

@author: laurent
"""


from dataset import generateData
import numpy as np

from scipy.spatial.distance import cdist
from scipy.stats import f as FisherDist

import matplotlib.pyplot as plt

import argparse as args

if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Plot and print a few stochastic informations about the training set')
	
	parser.add_argument("--trainsetsize", type=int, default = 10000, help="Number of samples to generate in the training set.")
	parser.add_argument("--trainsetobsprob", type=float, default = 0.2, help="probability of getting a measure for each item in an output vector for the train set.")
	parser.add_argument("--testsetsize", type=int, default = 5000, help="Number of samples to generate in the test set.")
	
	args = parser.parse_args()
	plt.rcParams["font.family"] = "serif"
	
	ret = generateData(args.trainsetsize, nTestPoints = args.testsetsize, mesThreshold = 1. - args.trainsetobsprob)
	trainInputs, trainOutputs, testInputs, testOutputs, trainMeasured, rReproject = ret
	
	normTestInputs = testInputs.T/np.std(testInputs.T, axis=0, keepdims=True)
	normTestOutputs = testOutputs.T/np.std(testOutputs.T, axis=0, keepdims=True)
	
	normDistMatIn = cdist(normTestInputs, normTestInputs, 'sqeuclidean')
	normDistMatOut = cdist(normTestOutputs, normTestOutputs, 'sqeuclidean')
	
	indices =  np.tril_indices(args.testsetsize, -1)
	
	vout = normDistMatOut[indices]
	vin = normDistMatIn[indices]
	vals = vout/vin
	
	print(np.min(vout), np.mean(vout), np.max(vout))
	print(np.min(vin), np.mean(vin), np.max(vin))
	print(np.min(vals), np.mean(vals), np.max(vals))
	
	fig, ax = plt.subplots()

	num_bins = 100
	# the histogram of the relative distance data
	n, bins, patches = ax.hist(vals, num_bins, density=True, label='Observed histogram')
	x = np.linspace(1e-2,np.max(bins), 2000)
	ax.plot(x, FisherDist.pdf(x, 9, 9),'--', color="orange", label='Fisher pdf')
	
	ax.set_xlabel('Relative distance squared')
	ax.set_ylabel('Probability density')
	#ax.set_xscale('log')
	ax.set_yscale('log')
	plt.legend()
	
	plt.show()