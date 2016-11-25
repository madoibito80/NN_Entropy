#	Neural Network Layer Entropy
import numpy as np
import chainer
import chainer.functions as F
from chainer import optimizers
import matplotlib.pyplot as plt


class mnist:

	fp1 = open("./train-images-idx3-ubyte","rb")
	fp2 = open("./train-labels-idx1-ubyte","rb")

	magic = sum([ord(fp1.read(1)) * pow(256,3-i) for i in range(4)])
	mnist_num = sum([ord(fp1.read(1)) * pow(256,3-i) for i in range(4)])
	img_row = sum([ord(fp1.read(1)) * pow(256,3-i) for i in range(4)])
	img_col = sum([ord(fp1.read(1)) * pow(256,3-i) for i in range(4)])

	magic = sum([ord(fp2.read(1)) * pow(256,3-i) for i in range(4)])
	mnist_num = sum([ord(fp2.read(1)) * pow(256,3-i) for i in range(4)])



	@classmethod
	def img(self):

		vector = np.ndarray((self.img_row,self.img_col))

		for y in range(self.img_row):
			for x in range(self.img_col):
				vector[y,x] = ord(self.fp1.read(1))

		vector /= 255

		return vector


	@classmethod
	def label(self):

		label = ord(self.fp2.read(1))

		return label





def entropy(layer):

	layer = layer.data[0].copy()
	dim = np.prod(np.array(layer.shape))
	layer = layer.reshape(dim)
	p = np.array([layer[int(np.random.rand()*dim)] for i in range(10)])
	p = np.exp(p)/np.sum(np.exp(p))
	e = -np.sum(p*np.log2(p))

	return e




def forward(model, x, i):

	h1 = F.max_pooling_2d(F.relu(model.conv1(x)),2)
	h2 = F.max_pooling_2d(F.relu(model.conv2(h1)),2)
	h3 = F.max_pooling_2d(F.relu(model.conv3(h2)),2)
	h4 = F.relu(model.fully1(h3))
	y = model.fully2(h4)

	t = int(i/par)
	x_e[t] += entropy(x)
	h1_e[t] += entropy(h1)
	h2_e[t] += entropy(h2)
	h3_e[t] += entropy(h3)
	h4_e[t] += entropy(h4)
	y_e[t] += entropy(y)
	
	return y




def main():

	global par
	par = 100

	dn = 10000

	global x_e, h1_e, h2_e, h3_e, h4_e, y_e
	k = int(dn/par)
	x_e = np.zeros(k)
	h1_e = np.zeros(k)
	h2_e = np.zeros(k)
	h3_e = np.zeros(k)
	h4_e = np.zeros(k)
	y_e = np.zeros(k) 


	model = chainer.FunctionSet(
		conv1 = chainer.links.Convolution2D(1,32,5),
		conv2 = chainer.links.Convolution2D(32,64,4),
		conv3 = chainer.links.Convolution2D(64,128,3),
		fully1 = F.Linear(512,256),
		fully2 = F.Linear(256,10)
	)


	optimizer = optimizers.SGD()
	optimizer.setup(model)



	for i in range(dn):

		print i

		data = mnist.img()
		label = mnist.label()

		x = chainer.Variable(data.reshape(1,1,28,28).astype(np.float32))
		t = chainer.Variable(np.array([label]).reshape(1).astype(np.int32))
		y = forward(model, x, i)

		optimizer.zero_grads()
		loss = F.softmax_cross_entropy(y,t)
		loss.backward()
		optimizer.update()

	print "plotting..."

	x = (np.arange(k)+1)*par
	plt.plot(x,x_e/par,label="x")
	plt.plot(x,h1_e/par,label="h1")
	plt.plot(x,h2_e/par,label="h2")
	plt.plot(x,h3_e/par,label="h3")
	plt.plot(x,h4_e/par,label="h4")
	plt.plot(x,y_e/par,label="y")

	plt.legend(loc="lower left")
	plt.xlabel("learned data")
	plt.ylabel("entropy")
	plt.savefig("./fig01.png")
	plt.show()


main()

