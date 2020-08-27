# -*- coding: utf-8 -*-

import numpy as np
from numpynet.layer import Dense, ELU, ReLU, SoftmaxCrossEntropy
from numpynet.function import Softmax
from numpynet.utils import Dataloader, one_hot_encoding, load_MNIST, save_csv
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

IntType = np.int64
FloatType = np.float64

#Decaying learning rate every 40 epochs
def decay_learning_rate(lr,epochss):
	if epochss > 160:
		lr = 0.02
	return lr


class Model(object):
	"""Model Your Deep Neural Network
	"""
	def __init__(self, input_dim, output_dim):
		"""__init__ Constructor

		Arguments:
			input_dim {IntType or int} -- Number of input dimensions
			output_dim {IntType or int} -- Number of classes
		"""
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.loss_fn = SoftmaxCrossEntropy(axis=0)
		self.build_model()

	def build_model(self):
		"""build_model Build the model using numpynet API
		"""
		# TODO: Finish this function
		self.Dense1 = Dense(self.input_dim,256)
		self.elu1 = ELU(alpha=0.9)
		self.Dense2 = Dense(256,128)
		self.elu2 = ELU(alpha=0.9)
		self.Dense3 = Dense(128,self.output_dim)
		self.elu3 = ELU(alpha=0.9)

		# raise NotImplementedError

	def __call__(self, X):
		"""__call__ Forward propogation of the model

		Arguments:
			X {np.ndarray} -- Input batch

		Returns:
			np.ndarray -- The output of the model. 
				You can return the logits or probits, 
				which depends on the way how you structure 
				the code.
		"""
		# TODO: Finish this function
		self.out_d1 = self.Dense1(X)
		self.out_e1 = self.elu1(self.out_d1)
		self.out_d2 = self.Dense2(self.out_e1)
		self.out_e2 = self.elu2(self.out_d2)
		self.out_d3 = self.Dense3(self.out_e2)
		self.out_e3 = self.elu3(self.out_d3)


		# return Softmax(self.out_e3)
		return self.out_e3
		# raise NotImplementedError

	def bprop(self, logits, labels, istraining=True):
		"""bprop Backward propogation of the model

		Arguments:
			logits {np.ndarray} -- The logits of the model output, 
				which means the pre-softmax output, since you need 
				to pass the logits into SoftmaxCrossEntropy.
			labels {np,ndarray} -- True one-hot lables of the input batch.

		Keyword Arguments:
			istraining {bool} -- If False, only compute the loss. If True, 
				compute the loss first and propagate the gradients through 
				each layer. (default: {True})

		Returns:
			FloatType or float -- The loss of the iteration
		"""

		# TODO: Finish this function
		self.logits,self.labels = logits,labels
		loss = self.loss_fn(self.logits, self.labels)
		if istraining==False:
			return loss
		#Backprop
		grad_softm_crossent = self.loss_fn.bprop()
		grad3 = grad_softm_crossent*self.elu3.bprop()
		grad2 = self.Dense3.bprop(grad3)*self.elu2.bprop()
		grad1 = self.Dense2.bprop(grad2)*self.elu1.bprop()
		grad0 = self.Dense1.bprop(grad1)
		return loss
		# raise NotImplementedError

	def update_parameters(self, lr):
		"""update_parameters Update the parameters for each layer.

		Arguments:
			lr {FloatType or float} -- The learning rate
		"""
		# TODO: Finish this function

		self.Dense3.update(lr)
		self.Dense2.update(lr)
		self.Dense1.update(lr)
		# raise NotImplementedError

import time

def train(model,
		  train_X,
		  train_y,
		  val_X,
		  val_y,
		  max_epochs=20,
		  lr=1e-3,
		  batch_size=16,
		  metric_fn=accuracy_score,
		  **kwargs):
	"""train Train the model

	Arguments:
		model {Model} -- The Model object
		train_X {np.ndarray} -- Training dataset
		train_y {np.ndarray} -- Training labels
		val_X {np.ndarray} -- Validation dataset
		val_y {np.ndarray} -- Validation labels

	Keyword Arguments:
		max_epochs {IntType or int} -- Maximum training expochs (default: {20})
		lr {FloatType or float} -- Learning rate (default: {1e-3})
		batch_size {IntType or int} -- Size of each mini batch (default: {16})
		metric_fn {function} -- Metric function to measure the performance of 
			the model (default: {accuracy_score})
	"""
	# TODO: Finish this function
	train_dataloader = Dataloader(X=train_X, y=train_y, batch_size=32, shuffle=True)
	val_dataloader = Dataloader(val_X, val_y,batch_size=32, shuffle=True) 

	train_acc,train_loss = [],[]
	val_acc,val_loss = [],[]
	for epoch in range(max_epochs):
		lr_decay = decay_learning_rate(lr,epoch)
		train_loss_single,correct_train = 0,0
		# train_acc = []
		for i,(inputs,labels) in enumerate(train_dataloader):
			labels_one_hot = one_hot_encoding(labels,num_class=10)
			output = model(inputs)
			loss_train = model.bprop(output,labels_one_hot)
			model.update_parameters(lr = lr_decay)

			
			pred = np.array([np.argmax(output[i]) for i in range(output.shape[0])]).reshape(-1,1)				
			train_loss_single += loss_train
			correct_train += np.ceil(accuracy_score(pred,labels)*batch_size)
		train_acc_single = correct_train/len(train_dataloader.y)
		train_acc.append(train_acc_single)
		train_loss.append(train_loss_single)
		print("Epoch: ", epoch+1, "Training Accuracy:", 100*train_acc_single,"%", "Loss:", train_loss_single / len(train_dataloader.y), "Learning Rate:", lr_decay)

		if (epoch+1)%2 == 0:
			#get validation loss and validation accuracy every two epochs
			val_loss_single,correct_val = 0,0
			for i,(inputs,labels) in enumerate(val_dataloader):
				labels_one_hot = one_hot_encoding(labels,num_class=10)
				output = model(inputs)   
				loss_val = model.bprop(output,labels_one_hot,istraining=False)
				
				pred = np.array([np.argmax(output[i]) for i in range(output.shape[0])]).reshape(-1,1)
				val_loss_single += loss_val
				correct_val += np.ceil(accuracy_score(pred,labels)*batch_size)
			val_acc_single = correct_val/len(val_dataloader.y)
			val_acc.append(val_acc_single)
			val_loss.append(val_loss_single)			
			print("Epoch: ", epoch+1, "Validation Accuracy:", 100*val_acc_single,"%", "Validation Loss:", val_loss_single / len(val_dataloader.y))
	return model, train_acc, train_loss, val_acc, val_loss

	# raise NotImplementedError


def inference(model, X, y=None, batch_size=16, metric_fn=accuracy_score, **kwargs):
	"""inference Run the inference on the given dataset

	Arguments:
		model {Model} -- The Neural Network model
		X {np.ndarray} -- The dataset input
		y {np.ndarray} -- The sdataset labels

	Keyword Arguments:
		metric {function} -- Metric function to measure the performance of the model 
			(default: {accuracy_score})

	Returns:
		tuple of (float, float): A tuple of the loss and accuracy
	"""
	# TODO: Finish this function
	test_dataloader = Dataloader(X, y=None,batch_size=32, shuffle=False)
	val_dataloader = Dataloader(X, y,batch_size=32, shuffle=True)

	print(len(test_dataloader.X))

	test_pred,val_pred = np.empty([1, 1]),np.empty([1, 1])
	if y is None:
		for i,(inputs) in enumerate(test_dataloader):
			output = model(inputs)   

			pred = np.array([np.argmax(output[i]) for i in range(output.shape[0])]).reshape(-1,1)
			test_pred = np.concatenate((test_pred, pred), axis=0)
		return test_pred[1:,:]
	# raise NotImplementedError
	else:
		val_loss,correct_val = 0,0
		for i,(inputs,labels) in enumerate(val_dataloader):
			labels_one_hot = one_hot_encoding(labels,num_class=10)
			output = model(inputs)   
			loss_val = model.bprop(output,labels_one_hot,istraining=False)
			

			pred = np.array([np.argmax(output[i]) for i in range(output.shape[0])]).reshape(-1,1)
			val_pred = np.concatenate((val_pred, pred), axis=0)
			val_pred = val_pred[1:,:]
			val_loss += loss_val
			correct_val += np.ceil(accuracy_score(pred,labels)*batch_size)
		val_acc = correct_val / len(val_dataloader.y)

		return val_acc, val_loss, val_pred


def main():
	print('loading data #####')
	train_X, train_y = load_MNIST(path ='dataset/',name="train")
	val_X, val_y = load_MNIST(path = 'dataset/', name="val")
	test_X = load_MNIST(path = 'dataset/', name="test")

	one_hot_train_y = one_hot_encoding(train_y)    


	test_loss, test_acc = None, None
	print('loading data complete #####')

	# TODO: 1. Build your model
	# TODO: 2. Train your model with training dataset and
	#       validate it  on the validation dataset
	# TODO: 3. Test your trained model on the test dataset
	#       you need have at least 95% accuracy on the test dataset to receive full scores

	# Your code starts here

	#NOTE: WE HAVE PROVIDED A SKELETON FOR THE MAIN FUNCTION. FEEL FREE TO CHANGE IT AS YOU WISH, THIS IS JUST A SUGGESTED FORMAT TO HELP YOU.

	batchSize = 32
	learningRate = 0.2 
	model = Model(input_dim = 784,output_dim = 10)
	model.build_model()
	print('Model built #####')
	model, train_acc, train_loss, val_acc, val_loss = train(model, train_X, train_y, val_X, val_y, max_epochs=200, lr=learningRate, batch_size=batchSize, metric_fn=accuracy_score)
	print('Training complete #####')

	# Plot of train and val accuracy vs iteration
	plt.figure(figsize=(10,7))
	plt.ylabel('Accuracy')
	plt.xlabel('Number of iterations')
	plt.title('Accuracy vs number of iterations')
	plt.plot(np.linspace(0,199,200), train_acc, label = 'Train accuracy across iterations')
	plt.plot(np.linspace(0,198,100), val_acc, label = 'Val accuracy across iterations')
	plt.legend(loc = 'upper right')
	plt.show()

	# Plot of train and val loss vs iteration
	plt.figure(figsize=(10,7))
	plt.ylabel('Loss')
	plt.xlabel('Number of iterations')
	plt.title('Loss vs number of iterations')
	plt.plot(np.linspace(0,199,200), train_loss, label = 'Train loss across iterations')
	plt.plot(np.linspace(0,198,100), val_loss, label = 'Val loss across iteration')
	plt.legend(loc='upper right')
	plt.show()

	# Implement inference such that you predict the labels and also evaluate val_accuracy and loss if true labels are provided
	val_acc, val_loss, val_pred = inference(model, val_X, val_y, batch_size = batchSize)

	# Inference on test dataset without labels

	#Implement inference function so that you can return the test prediction output and save it in test_pred. You are allowed to create a different function to generate just the predicted labels.
	test_pred = inference(model, test_X, test_y=None, batch_size = batchSize).reshape(-1, 1)
	print(test_pred.shape)
	save_csv(test_pred)
	# Your code ends here

	print("Validation loss: {0}, Validation Acc: {1}%".format(val_loss, 100 * val_acc))
	if val_acc > 0.95:
		print("Your model is well-trained.")
	else:
		print("You still need to tune your model")


if __name__ == '__main__':
	main()
