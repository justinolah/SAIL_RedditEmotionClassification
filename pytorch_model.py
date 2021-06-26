import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from helpers import *
from fasttext_model import *
from learn import *

class Feedforward(torch.nn.Module):
	def __init__(self, input_size, hidden_size):
		super(Feedforward, self).__init__()
		self.input_size = input_size
		self.hidden_size  = hidden_size
		self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
		self.relu = torch.nn.ReLU()
		self.fc2 = torch.nn.Linear(self.hidden_size, 1)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
		hidden = self.fc1(x)
		relu = self.relu(hidden)
		output = self.fc2(relu)
		output = self.sigmoid(output)
		return output

class MultilayerPerceptron(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(MultilayerPerceptron, self).__init__()
		self.fc1 = nn.Linear(input_dim, hidden_dim)
		self.fc2 = nn.Linear(hidden_dim, output_dim)
		self.sigmoid = torch.nn.Sigmoid()
		self.relu = torch.nn.ReLU()

	def forward(self, x_in):
		hiden = self.fc1(x_in)
		intermediate = self.relu(hiden)
		output = self.fc2(intermediate)
		output = self.sigmoid(output)
		return output

class GoEmotionsDatasetFasttext(Dataset):
    def __init__(self, data, emotions, maxSentenceLength=33):
        self.emotions = emotions
        self.maxSentenceLength = maxSentenceLength
        ft, wordVecLength = getFasttextModel()
        self.ft = ft
        self.wordVecLength = wordVecLength
        self.data = torch.Tensor(data.text.apply(cleanTextForEmbedding).apply(lambda x: getSentenceVectorPadded(x, ft, maxSentenceLength, wordVecLength)))
        self.labels = torch.Tensor(data.labels.apply(lambda x: getYMatrix(x,len(emotions))))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

def main():
	emotions = getEmotions()
	ft, wordVecLength = getFasttextModel()

	#parameters
	maxSentenceLength = 33
	epochs = 2
	input_dim = maxSentenceLength * wordVecLength
	hidden_dim = 50
	output_dim = len(emotions)
	lr = 1e-4

	#get data
	train = getTrainSet()
	test = getTestSet()

	#
	dataset = GoEmotionsDatasetFasttext(train, emotions, maxSentenceLength=33)

	#x_train = torch.Tensor(train.text.apply(cleanTextForEmbedding).apply(lambda x: getSentenceVectorPadded(x, ft, maxSentenceLength, wordVecLength)))
	#x_test = torch.Tensor(test.text.apply(cleanTextForEmbedding).apply(lambda x: getSentenceVectorPadded(x, ft, maxSentenceLength, wordVecLength)))

	#y_train = torch.Tensor(train.labels.apply(lambda x: getYMatrix(x,len(emotions))))
	#y_test = torch.Tensor(test.labels.apply(lambda x: getYMatrix(x,len(emotions))))

	#pytorch model
	torch.manual_seed(42)
	mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)

	optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
	loss_function = nn.CrossEntropyLoss()

	trainloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True, num_workers=1)

	for epoch in range(epochs):
		print("Epoch:", epoch)
		current_loss = 0.0

		for i, data in enumerate(trainloader):
			inputs, labels = data
			optimizer.zero_grad()
			outputs = mlp(inputs)

			loss = loss_function(outputs, labels)
			loss.backward()
			optimizer.step()

			current_loss += loss.item()

			if i % len(trainloader) == len(trainloader) - 1:
				print("Loss:", current_loss)
				current_loss = 0.0



if __name__ == "__main__":
	main()