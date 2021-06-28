import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from helpers import *
from fasttext_model import *
from learn import *

class MultilayerPerceptron(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MultilayerPerceptron, self).__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, output_dim),
			nn.Sigmoid(),
		)

	def forward(self, x_in):
		 return self.layers(x_in)

class GoEmotionsDatasetFasttext(Dataset):
	def __init__(self, data, emotions, ft=None, wordVecLength=300, maxSentenceLength=33):
		if ft is None:
			ft, wordVecLength = getFasttextModel()
		self.emotions = emotions
		self.maxSentenceLength = maxSentenceLength
		self.ft = ft
		self.wordVecLength = wordVecLength
		self.data = torch.Tensor(data.text.apply(cleanTextForEmbedding).apply(lambda x: getSentenceVectorPadded(x, ft, maxSentenceLength, wordVecLength)))
		self.labels = torch.Tensor(data.labels.apply(lambda x: getYMatrix(x,len(emotions))))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

class GoEmotionsDatasetGlove(Dataset):
	def __init__(self, data, emotions, gloveMap=None, wordVecLength=100, maxSentenceLength=33):
		if gloveMap is None:
			gloveMap, wordVecLength = getGloveMap()
		self.emotions = emotions
		self.maxSentenceLength = maxSentenceLength
		self.gloveMap = gloveMap
		self.wordVecLength = wordVecLength
		self.data = torch.Tensor(data.text.apply(cleanTextForEmbedding).apply(lambda x: getGloveVector(x, gloveMap, maxSentenceLength, wordVecLength)))
		self.labels = torch.Tensor(data.labels.apply(lambda x: getYMatrix(x,len(emotions))))

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]

def trainNN(model, trainloader, optimizer, loss_fn):
	counter = 0
	train_running_loss = 0.0

	for i, batch in enumerate(trainloader):
		counter += 1
		inputs, labels = batch
		optimizer.zero_grad()
		outputs = model(inputs)

		#print("inputs:", inputs)
		#print("labels:", labels)
		#print("outputs:", outputs)

		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	return train_running_loss / counter

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	#ft, wordVecLength = getFasttextModel()
	gloveMap, wordVecLength = getGloveMap()

	#parameters
	maxSentenceLength = 33
	epochs = 40
	input_dim = maxSentenceLength * wordVecLength
	hidden_dim = 2000
	output_dim = len(emotions)
	batch_size = 20
	threshold = 0.5
	lr = 1e-4
	filename = "mlp"

	#get data
	train = getTrainSet()
	test = getTestSet()

	print("Creating dataset...")
	#dataset = GoEmotionsDatasetFasttext(train, emotions, ft=ft, wordVecLength=wordVecLength, maxSentenceLength=33)
	#testset = GoEmotionsDatasetFasttext(test, emotions, ft=ft, wordVecLength=wordVecLength, maxSentenceLength=33)
	dataset = GoEmotionsDatasetGlove(train, emotions, gloveMap=gloveMap, wordVecLength=wordVecLength, maxSentenceLength=33)
	testset = GoEmotionsDatasetGlove(test, emotions, gloveMap=gloveMap, wordVecLength=wordVecLength, maxSentenceLength=33)
	print("Done\n")

	#pytorch model
	print("Training NN...")
	torch.manual_seed(42)
	mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)

	optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
	loss_fn= nn.BCELoss() #nn.BCEWithLogitsLoss() #nn.CrossEntropyLoss() #todo weights

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=1)

	#train model
	mlp.train()
	train_loss = []
	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss = trainNN(mlp, trainloader, optimizer, loss_fn)
		train_loss.append(epoch_loss)
		print("Loss:", epoch_loss, "\n")

	print("Training complete\n")

	#learning curve 
	plt.figure(figsize=(10, 7))
	plt.plot(train_loss, color='b', label='Training loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.legend()
	plt.savefig('plots/learningcurve.pdf')

	#Testing metrics
	mlp.eval()
	data, labels = testset[:]

	outputs = mlp(data)
	prediction = (outputs > threshold).int()

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
	multilabel_confusion_matrix(np.array(labels), outputs.detach().numpy(), emotions, top_x=3, filename=filename)

if __name__ == "__main__":
	main()