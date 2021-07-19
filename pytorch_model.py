import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchtext
from torchtext.vocab import GloVe, FastText
from torchtext.data import Field, Dataset, Example, Iterator, BucketIterator
from torchtext.data.utils import get_tokenizer
import pandas as pd
from sklearn.metrics import f1_score
from helpers import *
from learn import *
from loss_functions import *

class MultilayerPerceptron(nn.Module):
	def __init__(self, embedding, embedding_dim, input_dim, hidden_dim1, hidden_dim2, output_dim, dropout=0):
		super(MultilayerPerceptron, self).__init__()
		self.embedding = embedding
		if hidden_dim2 != 0:
			self.layers = nn.Sequential(
				nn.Dropout(p=dropout),
				nn.Linear(input_dim * embedding_dim, hidden_dim1),
				nn.ReLU(),
				nn.Dropout(p=dropout),
				nn.Linear(hidden_dim1, hidden_dim2),
				nn.ReLU(),
				nn.Linear(hidden_dim2, output_dim),
			)
		elif hidden_dim1 != 0:
			self.layers = nn.Sequential(
				nn.Dropout(p=dropout),
				nn.Linear(input_dim * embedding_dim, hidden_dim1),
				nn.ReLU(),
				nn.Linear(hidden_dim1, output_dim),
			)
		else:
			self.layers = nn.Sequential(
				nn.Linear(input_dim * embedding_dim, output_dim),
			)
	def forward(self, x):
		features = self.embedding[x].reshape(x.size()[0], -1)
		return self.layers(features)

class GoEmotionsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, fields: list, numEmotions: int):
        super(GoEmotionsDataset, self).__init__(
            [
                Example.fromlist(list(r), fields) 
                for i, r in data.iterrows()
            ], 
            fields
        )
        #self.balancedClassWeights = len(data.labels) / (numEmotions * torch.sum(torch.tensor(data.labels),0))
        #self.posWeights = torch.div((len(data.labels) - torch.sum(torch.tensor(data.labels),0)), 2.5 * torch.sum(torch.tensor(data.labels),0), rounding_mode='floor')

def makeDataset(data, emotions, text_field):
	data.labels = data.labels.apply(lambda x: getYMatrix(x,len(emotions)))
	labels_field = Field(sequential=False, use_vocab=False)
	return GoEmotionsDataset(
	    data=data, 
	    fields=(
	        ('text', text_field),
	        ('labels', labels_field)
	    ),
	    numEmotions = len(emotions)
	)

def trainNN(model, trainloader, devData, devLabels, optimizer, loss_fn, threshold, device):
	counter = 0
	train_running_loss = 0.0
	sigmoid = nn.Sigmoid()

	model.train()

	allTargets = []
	allPredictions = []

	for i, batch in enumerate(trainloader):
		counter += 1
		inputs = batch.text.T
		labels = batch.labels.float()
		allTargets.append(labels.detach())

		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = model(inputs)
		allPredictions.append((outputs.cpu() > threshold).int().detach())

		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	trainLoss = train_running_loss / counter

	model.eval()
	devOutput = model(devData.to(device))
	devLoss = loss_fn(devOutput, devLabels.to(device)).item()
	devOutput = sigmoid(devOutput)

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
	maxSentenceLength = 31
	embedding_dim = 200
	epochs = 3
	input_dim = maxSentenceLength
	hidden_dim1 = 1000
	hidden_dim2 = 0
	droput = 0.5
	weight_decay = 0.0001
	lr_decay = 0.95
	output_dim = len(emotions)
	batch_size = 64
	threshold = 0.5
	lr = 1e-3
	balanced=False
	filename = "mlp"

	print("Loading glove..\n")
	rawEmbedding = GloVe(name='twitter.27B', dim=embedding_dim)
	#rawEmbedding = FastText(language='en')

	#get data
	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	#clean text
	train.text = train.text.apply(cleanTextForEmbedding)
	test.text= test.text.apply(cleanTextForEmbedding)
	dev.text = dev.text.apply(cleanTextForEmbedding)

	text_field = Field(
	    sequential=True,
	    tokenize='basic_english', 
	    fix_length=maxSentenceLength,
	    lower=True
	)

	#build vocab
	preprocessed_text = train.text.apply(
    	lambda x: text_field.preprocess(x)
	)

	#print("Max sentence length:", np.array([len(row) for row in preprocessed_text.to_list()]).max()) #gets max sentence length

	text_field.build_vocab(
    	preprocessed_text, 
    	vectors=rawEmbedding,
	)

	vocab = text_field.vocab

	#Make datasets
	dataset = makeDataset(train, emotions, text_field)
	testset = makeDataset(test, emotions, text_field)
	devset = makeDataset(dev, emotions, text_field)

	#pytorch model
	print("Training NN...")
	torch.manual_seed(42)
	mlp = MultilayerPerceptron(vocab.vectors.to(device), embedding_dim, input_dim, hidden_dim1, hidden_dim2, output_dim)
	mlp.to(device)

	optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
	trainloader = Iterator(dataset, batch_size)

	weights = None
	if balanced:
		weights = dataset.balancedClassWeights
	rank_w = torch.zeros(output_dim)
	total = 0.
	for i in range(output_dim):
		total += 1./(i+1)
		rank_w[i] = total

	loss_fn= nn.BCEWithLogitsLoss(pos_weight= 8*torch.ones(len(emotions)).to(device))
	#loss_fn = wlsep
	#loss_fn = lambda x,y,weights=weights : warp(x,y,rank_w,weights=weights)

	#train model
	mlp.train()
	trainLoss = []
	devLoss = []
	trainF1 = []
	devF1 = []
	devbatch = next(iter(Iterator(devset, len(devset))))
	devData = devbatch.text.T
	devLabels = devbatch.labels.float()

	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss, dev_loss, f1_train, f1_dev = trainNN(mlp, trainloader, devData, devLabels, optimizer, loss_fn, threshold, device)
		trainLoss.append(epoch_loss)
		devLoss.append(dev_loss)
		trainF1.append(f1_train)
		devF1.append(f1_dev)
		print("Training Loss:", epoch_loss)
		print("Dev Loss:", dev_loss)
		print("Training Macro F1:", f1_train)
		print("Dev Macro F1:", f1_dev, "\n")

		lr *= lr_decay
		optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)

	print("Training complete\n")

	bestEpochDevF1 = np.argmax(np.array(devF1))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")

	#learning curve 
	fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 9))
	ax1.plot(trainLoss, color='b', label='Training loss')
	ax1.plot(devLoss, color='r', label='Dev loss')
	fig.suptitle(f"{hidden_dim1}X{hidden_dim2}, LR:{lr}, BS:{batch_size}, Embedding Size: {embedding_dim}")
	ax1.set(xlabel='Epochs', ylabel="Loss")
	ax1.legend()
	ax2.plot(trainF1, color='b', label='Training Macro F1')
	ax2.plot(devF1, color='r', label='Dev Macro F1')
	ax2.set(xlabel='Epochs', ylabel="Macro F1")
	ax2.legend()
	fig.savefig('plots/learningcurve.png')

	#Testing metrics
	mlp.eval()
	sigmoid = nn.Sigmoid()
	testbatch = next(iter(Iterator(testset, len(devset))))
	data = testbatch.text.T
	labels = testbatch.labels.float()

	outputs = mlp(data.to(device))
	outputs = sigmoid(outputs)
	prediction = (outputs > threshold).int().cpu()

	accuracy = accuracy_score(labels, prediction)
	print("Subset Accuracy:", accuracy)
	print(classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=False))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")
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
