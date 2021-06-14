#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:11:37 2021

@author: laurent
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from torch.distributions.normal import Normal


class VIModule(nn.Module) :
	"""
	A mixin class to attach loss functions to layer. This is usefull when doing variational inference with deep learning.
	"""
	
	def __init__(self, *args, **kwargs) :
		super().__init__(*args, **kwargs)
		
		self._internalLosses = []
		self.lossScaleFactor = 1
		
	def addLoss(self, func) :
		self._internalLosses.append(func)
		
	def evalLosses(self) :
		t_loss = 0
		
		for l in self._internalLosses :
			t_loss = t_loss + l(self)
			
		return t_loss
	
	def evalAllLosses(self) :
		
		t_loss = self.evalLosses()*self.lossScaleFactor
		
		for m in self.children() :
			if isinstance(m, VIModule) :
				t_loss = t_loss + m.evalAllLosses()*self.lossScaleFactor
				
		return t_loss

class L2RegularizedLinear(VIModule, nn.Linear) :
	
	def __init__(self, 
			  in_features, 
			  out_features,
			  bias=True, 
			  wPriorSigma = 1., 
			  bPriorSigma = 1.,
			  bias_init_cst = 0.0) :
		
		super().__init__(in_features, 
					   out_features,
					   bias=bias)
		
		if bias:
			self.bias.data.fill_(bias_init_cst)
		
		self.addLoss(lambda s : 0.5*s.weight.pow(2).sum()/wPriorSigma**2)
		
		if bias :
			
			self.addLoss(lambda s : 0.5*s.bias.pow(2).sum()/bPriorSigma**2)


class MeanFieldGaussianFeedForward(VIModule) :
	"""
	A feed forward layer with a Gaussian prior distribution and a Gaussian variational posterior.
	"""
	
	def __init__(self, 
			  in_features, 
			  out_features, 
			  bias = True,  
			  groups=1, 
			  weightPriorMean = 0, 
			  weightPriorSigma = 1.,
			  biasPriorMean = 0, 
			  biasPriorSigma = 1.,
			  initMeanZero = False,
			  initBiasMeanZero = False,
			  initPriorSigmaScale = 0.01) :
		
		
		super(MeanFieldGaussianFeedForward, self).__init__()
		
		self.samples = {'weights' : None, 'bias' : None, 'wNoiseState' : None, 'bNoiseState' : None}
		
		self.in_features = in_features
		self.out_features = out_features
		self.has_bias = bias
		
		self.weights_mean = Parameter((0. if initMeanZero else 1.)*(torch.rand(out_features, int(in_features/groups))-0.5))
		self.lweights_sigma = Parameter(torch.log(initPriorSigmaScale*weightPriorSigma*torch.ones(out_features, int(in_features/groups))))
			
		self.noiseSourceWeights = Normal(torch.zeros(out_features, int(in_features/groups)), 
								   torch.ones(out_features, int(in_features/groups)))
		
		self.addLoss(lambda s : 0.5*s.getSampledWeights().pow(2).sum()/weightPriorSigma**2)
		self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['wNoiseState'].pow(2).sum() - s.lweights_sigma.sum())
		
		if self.has_bias :
			self.bias_mean = Parameter((0. if initBiasMeanZero else 1.)*(torch.rand(out_features)-0.5))
			self.lbias_sigma = Parameter(torch.log(initPriorSigmaScale*biasPriorSigma*torch.ones(out_features)))
			
			self.noiseSourceBias = Normal(torch.zeros(out_features), torch.ones(out_features))
			
			self.addLoss(lambda s : 0.5*s.getSampledBias().pow(2).sum()/biasPriorSigma**2)
			self.addLoss(lambda s : -self.out_features/2*np.log(2*np.pi) - 0.5*s.samples['bNoiseState'].pow(2).sum() - self.lbias_sigma.sum())
			
			
	def sampleTransform(self, stochastic=True) :
		self.samples['wNoiseState'] = self.noiseSourceWeights.sample().to(device=self.weights_mean.device)
		self.samples['weights'] = self.weights_mean + (torch.exp(self.lweights_sigma)*self.samples['wNoiseState'] if stochastic else 0)
		
		if self.has_bias :
			self.samples['bNoiseState'] = self.noiseSourceBias.sample().to(device=self.bias_mean.device)
			self.samples['bias'] = self.bias_mean + (torch.exp(self.lbias_sigma)*self.samples['bNoiseState'] if stochastic else 0)
		
	def getSampledWeights(self) :
		return self.samples['weights']
	
	def getSampledBias(self) :
		return self.samples['bias']
	
	def forward(self, x, stochastic=True) :
		
		self.sampleTransform(stochastic=stochastic)
		
		return nn.functional.linear(x, self.samples['weights'], bias = self.samples['bias'] if self.has_bias else None)
	

class VariationalInferenceModuleSparseMeasure(VIModule) :
	"""
		VariationalInferenceModule is the actual BNN used for the sparse measure case study.
	"""
	
	def __init__(self, inputSize = 9, outputsize = 9, weight_sigma = 1., bias_sigma = 10., p_mc_dropout = 0.5) :
		super().__init__()
		
		self.p_mc_dropout = p_mc_dropout
		
		dropoutIncrFactor = 1
		if self.p_mc_dropout is not None :
			dropoutIncrFactor = 10 if (1-self.p_mc_dropout) <= 0.1 else int(1/(1-self.p_mc_dropout))
			if dropoutIncrFactor == 0 :
				dropoutIncrFactor = 1
		
		#The first layers are point estimate layers...
		self.layer1 = L2RegularizedLinear(inputSize,48, wPriorSigma = weight_sigma, bPriorSigma = bias_sigma)
		self.layer2 = L2RegularizedLinear(48,24, wPriorSigma = weight_sigma, bPriorSigma = bias_sigma)
		self.layer3 = L2RegularizedLinear(24,6, wPriorSigma = weight_sigma, bPriorSigma = bias_sigma)
		self.layer4 = L2RegularizedLinear(6,12*dropoutIncrFactor, wPriorSigma = weight_sigma, bPriorSigma = bias_sigma)
		
		#...only the last layers have a complete variational distribution attached on their parameters.
		self.layer5 = MeanFieldGaussianFeedForward(12*dropoutIncrFactor,24*dropoutIncrFactor, weightPriorSigma = weight_sigma, biasPriorSigma = bias_sigma)
		self.layer6 = MeanFieldGaussianFeedForward(24*dropoutIncrFactor, outputsize, weightPriorSigma = weight_sigma, biasPriorSigma = bias_sigma)
		
	def forward(self, x, stochastic=True):
		
		d = x
		
		d = nn.functional.leaky_relu(self.layer1(d))
		d = nn.functional.leaky_relu(self.layer2(d))
		d = nn.functional.leaky_relu(self.layer3(d))
		
		d = nn.functional.leaky_relu(self.layer4(d))
		if self.p_mc_dropout is not None :
			d = nn.functional.dropout(d, p = self.p_mc_dropout, training=stochastic) #MC-Dropout
			
		d = nn.functional.leaky_relu(self.layer5(d, stochastic=stochastic))
		if self.p_mc_dropout is not None :
			d = nn.functional.dropout(d, p = self.p_mc_dropout, training=stochastic) #MC-Dropout
			
		d = self.layer6(d, stochastic=stochastic)
		
		return d
	

		
class outlierAwareSumSquareError(VIModule) :
	
	def __init__(self, inputSize = 9,
			     expected_n_outliers = 1, 
				 unconstrained_logitsPriorSigma = 50., 
				 expected_n_outliersSharpness = 3000.) :
		super().__init__()
		
		self.unconstrained_logits = Parameter(torch.zeros(1,inputSize), requires_grad=True)
		self.expected_n_outliers = expected_n_outliers
		
		self.addLoss(lambda s : 0.5*s.unconstrained_logits.pow(2).sum()/unconstrained_logitsPriorSigma**2)
		self.addLoss(lambda s : expected_n_outliersSharpness*(torch.sigmoid(s.unconstrained_logits).sum() - s.expected_n_outliers)**2)
		
	def forward(self, x, batchgain = None, sigma_in = 0.1, sigma_out = 5.) :
		
		batchGain = batchgain
		if batchGain is None :
			batchGain = x.shape[0]
			
		constrained_logits = torch.sigmoid(self.unconstrained_logits)
		l = torch.sum((1-constrained_logits)*(x/sigma_in)**2 + constrained_logits*(x/sigma_out)**2)
		l += batchGain*torch.sum(torch.log((1-constrained_logits)*(sigma_in)**2 + constrained_logits*(sigma_out)**2))
		
		return 0.5*l
	
	def outliersProbability(self) :
		
		return torch.sigmoid(self.unconstrained_logits)
		