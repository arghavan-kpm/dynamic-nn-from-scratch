import math
import random

class Neuron:

	weight = [];
	bias = 0.0;
	active = "";
	inputSize = 0;
	cnt = 0;
	def __init__(self, active, N):
		self.active = active;
		self.inputSize = N;
		self.weight = [];
		self.bias = random.uniform( -1.0 * math.sqrt(1.0 / float(N)) , 1.0 * math.sqrt(1.0 / float(N)) );


		for i in range(0, N):
			self.weight.append( random.uniform( -1.0 * math.sqrt(1.0 / float(N)) , 1.0 * math.sqrt(1.0 / float(N)) ) );


	def sigmoid(self,x):
		if x >= 5:
			return 1.0;
		if x <= -5:
			return 0.0;
		S = 1.0 / (1.0 + math.exp(-x)); 
		#print ("X is : "  + str(x) + " sigmoid is : " + str(S)) ;
		return S;

	def pureCalc(self,L):
		res = 0.0;
		
		for i in range(0, self.inputSize):
			res += L[i] * self.weight[i];

		res += self.bias;

		return res;

	def calc(self, L):
		res = self.pureCalc(L);

		if self.active == "sigmoid":
			return self.sigmoid( res );

		if self.active == "linear":
			return res;

	def train( self, alpha, rondWeight, rondBias ):     # rondWeight[] -> weight[] ra update mikonim

		for i in range(0, len(self.weight)):			# rondBias     -> bias ra update mikonim
			self.weight[i] -= alpha * float(rondWeight[i]);
		self.bias -= alpha * float(rondBias);

	def setWeight(self,weight):
		self.weight = weight;
		
	def setBias(self,bias):
		self.bias = bias;