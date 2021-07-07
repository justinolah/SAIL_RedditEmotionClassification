import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.metrics import f1_score
from helpers import *
from fasttext_model import *
from learn import *
from loss_functions import *

class MultilayerPerceptron(nn.Module):
	def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0):
		super(MultilayerPerceptron, self).__init__()
		if hidden_dim2 != 0:
			self.layers = nn.Sequential(
				nn.Linear(input_dim, hidden_dim1),
				nn.ReLU(),
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim1, hidden_dim2),
				nn.ReLU(),
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim2, output_dim),
				nn.Sigmoid(),
			)
		elif hidden_dim1 != 0:
			self.layers = nn.Sequential(
				nn.Linear(input_dim, hidden_dim1),
				nn.ReLU(),
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim1, output_dim),
				nn.Sigmoid(),
			)
		else:
			self.layers = nn.Sequential(
				nn.Linear(input_dim, output_dim),
				nn.Sigmoid(),
			)
	def forward(self, x):
		return self.layers(x)

class GoEmotionsDatasetFasttext(Dataset):
	def __init__(self, data, emotions, ft=None, wordVecLength=300, maxSentenceLength=33):
		if ft is None:
			ft, wordVecLength = getFasttextModel()
		self.emotions = emotions
		self.numEmotions = len(emotions)
		self.maxSentenceLength = maxSentenceLength
		self.ft = ft
		self.wordVecLength = wordVecLength
		self.data = torch.Tensor(data.text.apply(cleanTextForEmbedding).apply(lambda x: getSentenceVectorPadded(x, ft, maxSentenceLength, wordVecLength)))
		self.labels = torch.Tensor(data.labels.apply(lambda x: getYMatrix(x,self.numEmotions)))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

	def getBalancedClassWeights(self):
		return len(self.labels) / (self.numEmotions * torch.sum(self.labels,0))

class GoEmotionsDatasetGlove(Dataset):
	def __init__(self, data, emotions, gloveMap=None, wordVecLength=100, maxSentenceLength=33):
		if gloveMap is None:
			gloveMap, wordVecLength = getGloveMap()
		self.emotions = emotions
		self.numEmotions = len(emotions)
		self.maxSentenceLength = maxSentenceLength
		self.gloveMap = gloveMap
		self.wordVecLength = wordVecLength
		self.data = torch.Tensor(data.text.apply(cleanTextForEmbedding).apply(lambda x: getGloveVector(x, gloveMap, maxSentenceLength, wordVecLength)))
		self.labels = torch.Tensor(data.labels.apply(lambda x: getYMatrix(x,self.numEmotions)))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

	def getBalancedClassWeights(self):
		return len(self.labels) / (self.numEmotions * torch.sum(self.labels,0))

def trainNN(model, trainloader, devData, devLabels, optimizer, loss_fn, weights, threshold, device):
	counter = 0
	train_running_loss = 0.0

	model.train()

	allTargets = []
	allPredictions = []

	for i, (inputs, labels) in enumerate(trainloader):
		counter += 1
		allTargets.append(labels.detach())

		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		allPredictions.append((outputs.cpu() > threshold).int().detach())

		loss = loss_fn(outputs, labels)#, weights=weights)
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	trainLoss = train_running_loss / counter

	model.eval()
	devOutput = model(devData.to(device))
	devLoss = loss_fn(devOutput, devLabels.to(device)).item()

	allPredictions = np.concatenate(allPredictions)
	allTargets = np.concatenate(allTargets)

	f1_train = f1_score(allTargets, allPredictions, average='macro')
	f1_dev = f1_score(devLabels.detach(), (devOutput.cpu() > threshold).int().detach(), average='macro')

	return trainLoss, devLoss, f1_train, f1_dev

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	#parameters
	embedding = "glove"
	maxSentenceLength = 33
	epochs = 5
	hidden_dim1 = 200
	hidden_dim2 = 0
	droput = 0
	output_dim = len(emotions)
	batch_size = 100
	threshold = 0.5
	lr = 1e-3
	balanced=True
	filename = "mlp"

	if embedding == "fasttext":
		ft, wordVecLength = getFasttextModel()
	elif embedding == "glove":
		gloveMap, wordVecLength = getGloveMap()
	else:
		print("Error: Invalid Embedding")
		return

	input_dim = maxSentenceLength * wordVecLength

	#get data
	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	print("Creating dataset...")
	if embedding == "fasttext":
		dataset = GoEmotionsDatasetFasttext(train, emotions, ft=ft, wordVecLength=wordVecLength, maxSentenceLength=33)
		testset = GoEmotionsDatasetFasttext(test, emotions, ft=ft, wordVecLength=wordVecLength, maxSentenceLength=33)
		devset = GoEmotionsDatasetFasttext(dev, emotions, ft=ft, wordVecLength=wordVecLength, maxSentenceLength=33)
	elif embedding == "glove":
		dataset = GoEmotionsDatasetGlove(train, emotions, gloveMap=gloveMap, wordVecLength=wordVecLength, maxSentenceLength=33)
		testset = GoEmotionsDatasetGlove(test, emotions, gloveMap=gloveMap, wordVecLength=wordVecLength, maxSentenceLength=33)
		devset = GoEmotionsDatasetGlove(dev, emotions, gloveMap=gloveMap, wordVecLength=wordVecLength, maxSentenceLength=33)
	else:
		print("Error: Invalid Embedding")
		return
	print("Done\n")

	#pytorch model
	print("Training NN...")
	torch.manual_seed(42)
	mlp = MultilayerPerceptron(input_dim, hidden_dim1, hidden_dim2, output_dim)
	mlp.to(device)

	optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)

	weights = None
	if balanced:
		weights = dataset.getBalancedClassWeights()
	rank_w = torch.zeros(output_dim)
	total = 0.
	for i in range(output_dim):
		total += 1./(i+1)
		rank_w[i] = total

	loss_fn= nn.BCELoss()
	#loss_fn = wlsep
	#loss_fn = lambda x,y,weights=weights : warp(x,y,rank_w,weights=weights)

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

	#train model
	mlp.train()
	trainLoss = []
	devLoss = []
	trainF1 = []
	devF1 = []
	devData, devLabels = next(iter(torch.utils.data.DataLoader(devset, batch_size=len(devset))))

	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss, dev_loss, f1_train, f1_dev = trainNN(mlp, trainloader, devData, devLabels, optimizer, loss_fn, weights, threshold, device)
		trainLoss.append(epoch_loss)
		devLoss.append(dev_loss)
		trainF1.append(f1_train)
		devF1.append(f1_dev)
		print("Training Loss:", epoch_loss)
		print("Dev Loss:", dev_loss)
		print("Training Macro F1:", f1_train)
		print("Dev Macro F1:", f1_dev, "\n")

	print("Training complete\n")

	#learning curve 
	fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 9))
	ax1.plot(trainLoss, color='b', label='Training loss')
	ax1.plot(devLoss, color='r', label='Dev loss')
	fig.suptitle(f"Embedding:{embedding}, H1:{hidden_dim1}, H2:{hidden_dim2}, LR:{lr}, BS:{batch_size}, Balanced:{balanced}")
	ax1.set(xlabel='Epochs', ylabel="Loss")
	ax1.legend()
	ax2.plot(trainF1, color='b', label='Training Macro F1')
	ax2.plot(devF1, color='r', label='Dev Macro F1')
	ax2.set(xlabel='Epochs', ylabel="Macro F1")
	ax2.legend()
	fig.savefig('plots/learningcurve.png')

	#Testing metrics
	mlp.eval()
	data, labels = testset[:]

	outputs = mlp(data.to(device))
	prediction = (outputs > threshold).int().cpu()

	accuracy = accuracy_score(labels, prediction)
	print("Subset Accuracy:", accuracy)
	print(classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=False))
	report = classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=True)

	#export resuls to csv
	micro = list(report['micro avg'].values())
	micro.pop() 
	macro = list(report['macro avg'].values())
	macro.pop()
	scores = [accuracy, *micro, *macro]
	results = pd.DataFrame(data=[scores], columns=['accuracy', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'])
	results.to_csv("tables/" + filename + "_results.csv")

	#confusion matrix
	multilabel_confusion_matrix(np.array(labels), outputs.cpu().detach().numpy(), emotions, top_x=3, filename=filename)

if __name__ == "__main__":
	main()