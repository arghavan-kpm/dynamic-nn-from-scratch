from layer import *
import ast
import math

EPSILON = 1e-10

class Net:

	layers = [];
	layerOut = []; # output haye har layer
	opt = "";
	alpha = 0.0;
	
	def __init__(self, opt, size, active, regul, alpha): # size , tedad neuron haye hame laye ha ( input layer ham ) hast 
		self.opt = opt;
		self.alpha = alpha;

		for i in range(0, len(active)):
			self.layers.append( Layer(size[i+1], active[i], size[i], regul[i] ) );

	def GD_SGD(self,All):
		Err = 0.0;
		
		for i in range(0,len(All)):
			data = All[i];
			Err += self.lossFunc(self.calc(data[0]),data[1]);
		Err /= float(len(All));

		return Err;


	def calc( self,L ):
		layerOut = [];

		L = self.layers[0].L2Norm(L);
		layerOut.append(L);
		for i in range(0, len(self.layers)):				
			if i == len(self.layers) -1 :

				layerOut.append(self.layers[i].pureCalc(L));
			else:
				layerOut.append(self.layers[i].calc(L));
			L = self.layers[i].calc(L);	

		self.layerOut.append(layerOut);

		return L;


	def lossFunc(self, calcOut, char):
		
		Y = [];
		for i in range(0, self.layers[len(self.layers) - 1].size ):
			Y.append(0.0);
		Y[ ord(char) - ord('A')] = 1.0;

		Err = [];

		for i in range(0, len(calcOut)):
			Err.append( Y[i] * math.log(self.sigmoid(calcOut[i]) + 1e-9) + (1.0 - Y[i]) * math.log(1.0 - self.sigmoid(calcOut[i]) + 1e-9) );

		return -sum(Err) / len(Err);

	
	def sigmoid(self,x):
		if x >= 5:
			return 1;
		if x <= -5:
			return 0.0;
		S = 1.0 / (1.0 + math.exp(-x)); 
		#print ("X is : "  + str(x) + " sigmoid is : " + str(S)) ;
		return S;

	def sigmoidP(self,x):
		return self.sigmoid(x) * (1.0 - self.sigmoid(x));


	def compute(self, Y,Lout):
		res = [];

		for i in range(0, len(Y)):
			res.append( -1.0 * (Y[i] * (self.sigmoidP(Lout[i])) / (self.sigmoid(Lout[i])+EPSILON) + (Y[i] - 1.0) * (self.sigmoidP(Lout[i])) / (1.0 - (self.sigmoid(Lout[i])+EPSILON) ) ) / float(len(Y)) );
				

		return res;

	def rond(self, All):    # rond(error) / rond(Lout) baraye hame data haye train

		self.layerOut = []; # dar iter e jaDd pak shavad 
		resAll = [];
		for i in range(0, self.layers[len(self.layers) - 1].size ):
			resAll.append(0.0);

		Y = [];
		for j in range(0, self.layers[len(self.layers) - 1].size ):
			Y.append(0.0);
	
		for i in range(0 , len(All)):
		
			Y[ ord(All[i][1]) - ord('A')] = 1.0;

			s = self.calc (All[i][0]);  # tamam e khoruG haye e hame layer ha ra be eza e data set e dade shode hesab mikonim   vali   meghdar e khoruji fght Lout ra b eza e an data set return mikonim
			Lout = self.layerOut[i][len(self.layers)];

			computeFor_ith_data = self.compute(Y,Lout);

			for j in range(0, len(resAll)):
				resAll[j] += computeFor_ith_data[j];

			Y[ ord(All[i][1]) - ord('A')] = 0.0;

		for i in range(0, len(resAll)):
			resAll[i] /= float(len(resAll));

		return resAll;

	def matrixMult(self, A, B):

		res = [];
		for i in range(0,len(A)):
			t = [];
			for j in range(0,len(B[0])):
				Sum = 0.0;
				for k in range(0,len(A[0])):			
					Sum += float(A[i][k]) * float(B[k][j]);

				t.append(Sum);
			res.append(t);
		return res;

	def transpose(self,A):
		res = [];
		for i in range(0,len(A[0])):
			t = [];
			for j in range(0,len(A)):
				t.append(A[j][i]);
			res.append(t);
		return res;


	def backProp(self,Rond):
		
		W = [];
		B = [];

		avgLayerOut = [];

		for j in range(0,len(self.layerOut[0])): # len(layers[]) : 3
			t2 = [];
			for k in range(0, len(self.layerOut[0][j])): # 784 ya 100 ya 10
				t2.append(0.0);
			avgLayerOut.append(t2);	

		for i in range(0, len(self.layerOut)):
			for j in range(0, len(self.layerOut[i])):
				for k in range(0, len(self.layerOut[i][j])):
					avgLayerOut[j][k] += self.layerOut[i][j][k];


		for j in range(0, len(self.layerOut[i])):
				for k in range(0, len(self.layerOut[i][j])):
					avgLayerOut[j][k] /= float(len(self.layerOut));

		'''  avgLayerOut yek array az 3 ta array ast ba size haye 784 100 10 tayi k avg e 18700 data set ast  '''



		Rond = [Rond];
		for i in range(0, len(self.layers)):
			
			W.append( self.matrixMult( self.transpose(Rond), [ avgLayerOut[ len(self.layers) -1 - i ] ] ) );
			B.append( Rond );
			Rond = self.matrixMult( Rond , self.layers[ len(self.layers) -1 -i ].weight());

		W.reverse();
		B.reverse();
		return W, B;


	def train(self, All):
		
		Rond = self.rond(All);
		
		W , B = self.backProp(Rond);

		for i in range(0,len(self.layers)):
			self.layers[i].train(self.alpha,W[i],B[i][0]);

		return self.GD_SGD(All);

	def LoadNet(self,FileName):
		File = open(FileName,"r");
		Lines = File.readlines();
		File.close();
		Weights = [];
		cnt = 0;
		while cnt < len(Lines):
			self.layers[cnt // 2].setLayer( ast.literal_eval(Lines[cnt]) , ast.literal_eval(Lines[cnt + 1]));
			cnt += 2;

	def SaveNet(self,FileName):
		File = open(FileName,"w");
		for i in range(0,len(self.layers)):
			File.write(str(self.layers[i].weight()) + "\n");
			File.write(str(self.layers[i].bias()) + "\n");
		File.close();


		
