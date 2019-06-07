from Net import *
from PIL import Image
import os
import random


itr = 0;
def test( L ,net):

	c = ['A','B','C','D','E','F','G','H','I','J'];

	L = net.calc(L);
	
	maxInd = 0;
	for i in range(1,len(L)):
		if L[i] > L[maxInd] :
			maxInd = i;
	return c[maxInd];


def main():
	#     optimizer, size[],         actFuncs[],            reguls[],             learningRate
	#net = Net("SGD", [784, 500, 300, 100, 50, 10], ["linear", "linear", "linear",  "linear", "sigmoid"], ["nothing", "dropOut", "nothing", "nothing" , "nothing"], 0.1);
	net = Net("SGD", [784, 100, 10], ["linear", "sigmoid"], ["dropOut", "nothing"], 0.7);
	All = [];

	for filename in os.listdir(os.getcwd() +"/DataSet"):
		print(filename);

		for file in os.listdir("./DataSet/" + filename):

			if file.endswith(".png"):

				im = Image.open("./DataSet/" + filename + "/" + file) 
				pix = im.load()
				L = [];
				for x in range(0, 28):
					for y in range(0, 28):
						L.append( pix[x,y] );

				All.append( (L, filename) );   # L ha ra b hamrah e javabeshan dar A negah midarim

	print("******************************************************************TRAINING********************************************************************************");
	random.shuffle(All);
	
	cnt = 0
	Err = [];

	net.LoadNet("Net_bgd.net");

	if( net.opt == "SGD"):
		for j in range(0,itr):
			cnt = 0;
			for i in range(0,4000):
				err = net.train([All[cnt]]);
				if err < 0.4:
					net.alpha = 0.1;
				if i % 100 == 0:
					print("iter : " + str(i) + "\terror : " + str( err ) ) ;
				Err.append( err );
				cnt += 1;

	if( net.opt == "GD"):
		for i in range(0, itr):
			err = net.train(All[0:len(All) / 100]);
			if err < 0.4:
					net.alpha = 0.1;
			print("iter : " + str(i) + "\terror : " + str( err ) ) ;
			Err.append( err );
			cnt += 1;

	#print( Err );
	#net.SaveNet("Net_bgd.net");
	#net.SaveNet("Net_sgd.net");
	print("******************************************************************TESTING*********************************************************************************");
	
	cnt = 0;
	for i in range(5000, 5100):
		if test( All[i][0] , net) <> All[i][1] :
			cnt += 1;
		print("geuss is " + test( All[i][0] , net) + " and it's actually " + All[i][1]);
	print( " total : 100 , mistake : " + str(cnt));

main();