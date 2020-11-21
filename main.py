from opts import opts
from net import *
from PIL import Image
import os
import random


def test( L ,net):

	c = ['A','B','C','D','E','F','G','H','I','J'];
	L = net.calc(L);
	
	maxInd = 0;
	for i in range(1,len(L)):
		if L[i] > L[maxInd] :
			maxInd = i;
	return c[maxInd];


def main(opt):
	#     optimizer, size[],         actFuncs[],            reguls[],             learningRate
	#net = Net("GD", [784, 100, 10], ["linear", "sigmoid"], ["dropOut", "nothing"], 0.7);
	net = Net(opt.optimizer, opt.layers, opt.active_funcs, opt.regularizers, opt.lr)

	All = [];

	for foldername in os.listdir(opt.dataset_dir):
		folder_dir = os.path.join(opt.dataset_dir, foldername)
		for file in os.listdir(folder_dir):

			if file.endswith(".png"):

				im = Image.open(os.path.join(folder_dir, file)) 
				pix = im.load()
				L = [];
				for x in range(0, 28):
					for y in range(0, 28):
						L.append( pix[x,y] );

				All.append( (L, foldername) );   # L ha ra b hamrah e javabeshan dar ALL negah midarim

	print(f"#  dataset is loaded.")	
	print("#  /////////////////////////TRAINING STARTED////////////////////////////////");
	random.shuffle(All);
	
	cnt = 0
	Err = [];

	net.LoadNet(os.path.join('./weights', 'w.net'));

	if( net.opt == "SGD"):
		for j in range(0,opt.num_epochs):
			cnt = 0;
			for i in range(0, 200):
				err = net.train([All[cnt]]);
				if err < 0.4:
					net.alpha = 0.1;
				if i % 100 == 0:
					print("#  iter : " + str(i) + "\terror : " + str( err ) ) ;
				Err.append( err );
				cnt += 1;

	if( net.opt == "GD"):
		for i in range(0, opt.num_epochs):
			err = net.train(All[0:len(All) // 100]);
			if err < 0.4:
					net.alpha = 0.1;
			print("#  iter : " + str(i) + "\terror : " + str( err ) ) ;
			Err.append( err );
			cnt += 1;

	#net.SaveNet("w_new.net");
	print("#  ////////////////////////EVALUATION STARTED///////////////////////////////");
	
	cnt = 0;
	for i in range(0, 1000):
		if test( All[i][0] , net) != All[i][1] :
			cnt += 1;
		#print("geuss is " + test( All[i][0] , net) + " and it's actually " + All[i][1]);
	print( 
		f"#  for 1000 training data sample, accuracy is {100 - (cnt/1000 * 100)}",
		f"#",
		f"############################################################################",
		sep='\n'
		);

if __name__ == "__main__":
	opt = opts().parse()
	main(opt)