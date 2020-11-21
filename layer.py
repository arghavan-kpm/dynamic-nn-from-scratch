from neuron import *
import math
import random

prob = 0.5;

class Layer:
	neurons = [];
	size = 0;
	regul = "";

	def __init__(self, size, active, prevSize, regul):
		self.size = size;
		self.regul = regul;
		self.neurons = [];
		for i in range(0, size):
			self.neurons.append( Neuron(active, prevSize));

	
	def weight(self):
		w = [];
		for i in range(0, len(self.neurons)):
			w.append( self.neurons[i].weight );
		return w;

	def bias(self):
		b = [];
		for i in range(0,len(self.neurons)):
			b.append(self.neurons[i].bias);
		return b;

	def dropOut(self, L ):
		for i in range(0, len(L)):
			if random.uniform(0, 1) < prob:
				L[i] = 0.0;

		return L;


	def L2Norm(self, L ):
		summation = 0.0; 

		for i in range( 0 , len(L)):
			summation += L[i] ** 2;

		summation = math.sqrt(summation);

		for i in range(0, len(L)):
			L[i] = L[i] / summation;

		return L;


	def pureCalc(self,L):
		res = [];

		for i in range(0, self.size):
			
			res.append(self.neurons[i].pureCalc(L) );
		return res;

	def calc(self, L ):

		res = self.pureCalc(L);
		if self.regul == "dropOut":
			return self.dropOut(res);

		if self.regul == "L2Norm":
			return self.L2Norm(res);

		if self.regul == "nothing":
			return res;
			
	def train(self, alpha, neuronWs, neuronB):  # neuronWs[][] , neuronB[]
		for i in range(0,len(self.neurons)):
			self.neurons[i].train(alpha, neuronWs[i],neuronB[i]);

	def setLayer(self,weight,bias):
	
		for i in range(0,len(self.neurons)):
			self.neurons[i].setWeight(weight[i]);
			self.neurons[i].setBias(bias[i]);