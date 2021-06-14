#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 10:46:27 2021

@author: laurent
"""


from dataset import generateData
from viModel import VariationalInferenceModuleSparseMeasure as sparseMeasureBnn
from viModel import outlierAwareSumSquareError

import numpy as np
from scipy.stats import chi2 as Chi2Dist

import matplotlib.pyplot as plt

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import argparse as args

class sparseMeasureDataset(Dataset):

	def __init__(self, inputs, outputs, measured) :
		self.inputs = inputs.astype(np.float32)
		self.outputs = outputs.astype(np.float32)
		self.measured = measured.astype(np.float32)
		
	def __len__(self):
		return self.inputs.shape[1]

	def __getitem__(self, idx):
	
		sample = {'input': self.inputs[:,idx], 'output': self.outputs[:,idx], 'measured': self.measured[:,idx]}
		return sample
	
if __name__ == "__main__" :
	
	parser = args.ArgumentParser(description='Train the sparse measure BNN using variational inference')
	
	parser.add_argument("--trainsetsize", type=int, default = 1000, help="Number of samples to generate in the training set.")
	parser.add_argument("--trainsetobsprob", type=float, default = 0.3, help="probability of getting a measure for each item in an output vector for the train set.")
	parser.add_argument("--testsetsize", type=int, default = 1000, help="Number of samples to generate in the test set.")
	
	parser.add_argument('--nepochs', type=int, default = 2000, help="The number of epochs to train for")
	parser.add_argument('--nruntests', type=int, default = 100, help="The number of pass to use at test time for monte-carlo uncertainty estimation")
	parser.add_argument('--learningrate', type=float, default = 1e-2, help="The learning rate of the optimizer")
	#parser.add_argument('--numnetworks', type=int, default = 1, help="The number of networks to train in parallel")
	
	args = parser.parse_args()
	plt.rcParams["font.family"] = "serif"
	
	ret = generateData(args.trainsetsize, nTestPoints = args.testsetsize, mesThreshold = 1. - args.trainsetobsprob)
	trainInputs, trainOutputs, testInputs, testOutputs, trainMeasured, rReproject = ret
	n_obs = np.sum(trainMeasured)
	
	
	model = sparseMeasureBnn()
	outlierAwareSSE = outlierAwareSumSquareError()
	
	training_data = sparseMeasureDataset(trainInputs, trainOutputs, trainMeasured)
	
	train_dataloader = DataLoader(training_data, batch_size=50, shuffle=True)
	
	projectionKernel = torch.from_numpy((rReproject @ rReproject.T).astype(np.float32))
		
	paramsNN = [p for p in model.parameters()]
	paramsLoss = [p for p in outlierAwareSSE.parameters()]
	
	optimizer = Adam(paramsNN + paramsLoss, lr=args.learningrate)
	optimizer.zero_grad()
	
	#global parameters
	sigma_y = 0.1
	sigma_in = 0.1
	sigma_out = 5.
	sigma_plane = 1.
	lambda_dist = 1.
	
	#training loop
	print("Training:")
	for n in range(args.nepochs) :
		
		print("Epoch {}/{}:".format(n+1, args.nepochs))
		
		batchLen = len(train_dataloader)
		digitsBatchLen = len(str(batchLen))
		
		for batch_id, sampl in enumerate(train_dataloader) :
			
			inputs = sampl['input']
			outputs = sampl['output']
			measured = sampl['measured']
			
			batch_size = inputs.shape[0]
			minibatchscale = float(args.trainsetsize)/float(batch_size)
			
			pred = model(inputs)
			
			#Computing the different contribution to the loss
			#The different contribution to the loss are scaled to account for the size of the plate in the BBN
			weigthedSquareErrors = n_obs/measured.sum() * outlierAwareSSE(measured*(pred - outputs),
																		 batchgain = measured.sum()/9.,
																		 sigma_in = np.sqrt(sigma_y**2 + sigma_in**2), 
																		 sigma_out = np.sqrt(sigma_y**2 + sigma_out**2))
			
			projectionError = (float(args.trainsetsize)/batch_size) * 0.5*torch.sum(((pred - torch.matmul(pred, projectionKernel))/sigma_plane)**2) #torch.zeros(())
			
			indices = torch.tril_indices(batch_size, batch_size, offset=-1)
			distMatrInput = torch.sum((inputs[:,np.newaxis,:] - inputs[np.newaxis,:,:])**2, axis=-1, keepdim=False)
			distMatrOutput = torch.sum((pred[:,np.newaxis,:] - pred[np.newaxis,:,:])**2, axis=-1, keepdim=False)
			
			FStatistics = lambda_dist*(distMatrOutput[indices[0], indices[1]]/distMatrInput[indices[0], indices[1]])
			
			graphReg = (float(args.trainsetsize)/batch_size) * torch.sum( 8. * torch.log(1. + FStatistics) - 3.5 * torch.log(FStatistics) - 5.6) #offset by 5.6, which is approximatly log(1/Beta(4.5,4.5)), to avoid having constant that is too large in the printed loss.
			
			l = weigthedSquareErrors + projectionError + graphReg
			#add the VI losses.
			smartLossLosses = outlierAwareSSE.evalAllLosses()
			l += model.evalAllLosses() + smartLossLosses
			
			optimizer.zero_grad()
			l.backward()
			
			optimizer.step()
			
			print("\r", ("\tTrain step {"+(":0{}d".format(digitsBatchLen))+"}/{} error = {:.4f}, projection = {:.4f}, graph = {:.4f}, smartloss = {:.4f}").format(batch_id+1, 
						    batchLen, 
							weigthedSquareErrors.detach().cpu().item(),
							projectionError.detach().cpu().item(),
							graphReg.detach().cpu().item(),
							smartLossLosses.detach().cpu().item()), end="")
			
		print("")
		
	print("")
	print("Outlier detection logits:")
	print(outlierAwareSSE.outliersProbability().detach().cpu().numpy())
	print("")
		
	#different evaluation metrics
	print("Evaluation:")
	
	model.eval()
	with torch.no_grad():
		#accuracy
		inputs = torch.from_numpy(testInputs.astype(np.float32).T)
		outputs = torch.from_numpy(testOutputs.astype(np.float32).T)
			
		batch_size = inputs.shape[0]
		
		stochasticPreds = torch.zeros((batch_size, 9, args.nruntests))
		for n in np.arange(args.nruntests) :
			stochasticPreds[:,:,n] = model(inputs, stochastic=True)
		
		predictions = stochasticPreds.mean(axis=-1, keepdim=True)
		errors = outputs - predictions.squeeze(dim=-1)
		errors_np = errors.cpu().numpy()
			
		print("Root mean square error of average model: {}".format(np.sqrt(np.mean(errors_np**2))))
	
		fig, ax = plt.subplots()
	
		num_bins = 100
		# the histogram of the actual error distribution
		n, bins, patches = ax.hist(errors_np.flatten(), num_bins, density=True, label='Observed histogram')
		ax.set_xlabel('Actual prediction error')
		ax.set_ylabel('Probability density')
		
		covs = stochasticPreds - predictions
		
		covs = torch.matmul(covs, torch.transpose(covs, 1, 2))/(args.nruntests-1.)
		weigths = torch.linalg.inv(covs) #
		
		nssr = torch.matmul(errors[:,np.newaxis,:], torch.matmul(weigths, errors[:,:,np.newaxis]))
		nssr = nssr.cpu().numpy().flatten()
		nssr = np.sort(nssr)
		p_obs = np.linspace(1./nssr.size,1.0,nssr.size)
		p_pred = Chi2Dist.cdf(nssr, 9);
		
		plt.figure("Calibration curve for sparse measure model")
		plt.plot(p_pred, p_obs, label='Calibration curve')
		plt.plot([0,1],[0,1], 'k--', alpha=0.5, label='Ideal curve')
		plt.xlabel('Predicted probability')
		plt.ylabel('Observed probability')
		plt.axis('equal')
		plt.xlim([0,1])
		plt.ylim([0,1])
		plt.legend()
		
		plt.show()