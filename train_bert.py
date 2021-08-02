import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import BertModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader

from helpers import *
from learn import *

import wandb

wandb.init(project='SAILGoemotions', entity='justinolah')
config = wandb.config

seed_value=42
np.random.seed(seed_value)
torch.manual_seed(seed_value)
random.seed(seed_value) 
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class BERT_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(BERT_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		_, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
		out = self.fc(cls_hs)
		return out


def makeBERTDataset(data, tokenizer, max_length, emotions):
	data.labels = data.labels.apply(lambda x: getYMatrix(x,len(emotions)))

	tokens = tokenizer.batch_encode_plus(
		data.text.tolist(),
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])
	y = torch.tensor(data.labels.tolist())

	dataset = TensorDataset(seq, mask, y)
	return dataset

def trainNN(model, trainloader, devloader, optimizer, loss_fn, threshold, device):
	counter = 0
	train_running_loss = 0.0
	sigmoid = nn.Sigmoid()

	model.train()

	allTargets = []
	allPredictions = []

	for i, batch in enumerate(trainloader):
		counter += 1
		seq, mask, labels = batch
		allTargets.append(labels.detach())
		print(i)
		optimizer.zero_grad()
		output = model(seq.to(device), mask.to(device))
		allPredictions.append((output.cpu() > threshold).int().detach())

		loss = loss_fn(output, labels.float().to(device))
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	trainLoss = train_running_loss / counter

	model.eval()
	seq, mask, labels = next(iter(devloader))
	devOutput = model(seq.to(device), mask.to(device))
	devLoss = loss_fn(devOutput, labels.float().to(device)).item()
	devOutput = sigmoid(devOutput)

	allPredictions = np.concatenate(allPredictions)
	allTargets = np.concatenate(allTargets)

	f1_train = f1_score(allTargets, allPredictions, average='macro')
	f1_dev = f1_score(labels.detach(), (devOutput.cpu() > threshold).int().detach(), average='macro')

	return trainLoss, devLoss, f1_train, f1_dev

def main():
	epochs = 3
	batch_size = 64
	max_length = 128
	weight_decay = 0.0001
	lr_decay = 0.95
	threshold = 0.5
	lr = 1e-3
	init_lr = lr
	decay_start = 5
	filename = "bert"

	freeze_bert = True

	config.epochs = epochs
	config.batch_size = batch_size
	config.max_length = max_length
	config.weight_decay = weight_decay
	config.lr_decay = lr_decay
	config.lr = lr_decay
	config.decay_start = decay_start
	config.framework = "bert"

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	#make Datasets
	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	train_set = makeBERTDataset(train, tokenizer, max_length, emotions)
	test_set = makeBERTDataset(test, tokenizer, max_length, emotions)
	dev_set = makeBERTDataset(dev, tokenizer, max_length, emotions)

	trainloader = DataLoader(train_set, batch_size=batch_size)
	testloader = DataLoader(test_set, batch_size=len(dev_set))
	devLoader = DataLoader(dev_set, batch_size=len(dev_set))

	#initialize model
	bert = BertModel.from_pretrained('bert-base-uncased')
	if freeze_bert:
		for param in bert.parameters():
			param.requires_grad = False
	model = BERT_Model(bert, len(emotions))
	model = model.to(device)
	wandb.watch(model)

	loss_fn= nn.BCEWithLogitsLoss(pos_weight= 8*torch.ones(len(emotions)).to(device))

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	#train model
	model.train()
	trainLoss = []
	devLoss = []
	trainF1 = []
	devF1 = []

	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss, dev_loss, f1_train, f1_dev = trainNN(model, trainloader, devLoader, optimizer, loss_fn, threshold, device)
		trainLoss.append(epoch_loss)
		devLoss.append(dev_loss)
		trainF1.append(f1_train)
		devF1.append(f1_dev)
		print("Training Loss:", epoch_loss)
		print("Dev Loss:", dev_loss)
		print("Training Macro F1:", f1_train)
		print("Dev Macro F1:", f1_dev, "\n")

		wandb.log({"train_loss": epoch_loss})
		wandb.log({"dev_loss": dev_loss})
		wandb.log({"train_f1": f1_train})
		wandb.log({"dev_f1": f1_dev})

		if epoch == 0 or devF1[-1] > devF1[-2]:
			torch.save({
	            'epoch': epoch,
	            'model_state_dict': rnn.state_dict(),
	            'devF1' : devF1[-1],
	            }, "bert.pt")

		if epoch > decay_start:
			lr *= lr_decay
			optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

	print("Training complete\n")

	#learning curve 
	fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 9))
	ax1.plot(trainLoss, color='b', label='Training loss')
	ax1.plot(devLoss, color='r', label='Dev loss')
	fig.suptitle(f"Bert Arch. LR:{init_lr}, BS:{batch_size}")
	ax1.set(xlabel='Epochs', ylabel="Loss")
	ax1.legend()
	ax2.plot(trainF1, color='b', label='Training Macro F1')
	ax2.plot(devF1, color='r', label='Dev Macro F1')
	ax2.set(xlabel='Epochs', ylabel="Macro F1")
	ax2.legend()
	fig.savefig("plots/learningcurve_" + filename + ".png")

	#Testing metrics
	bestCheckpoint = torch.load("bert.pt")
	model.load_state_dict(bestCheckpoint['model_state_dict'])
	bestEpochDevF1 = bestCheckpoint['epoch']
	bestDevF1 = bestCheckpoint['devF1']
	print("Best Dev F1:", bestDevF1, "at epoch", bestEpochDevF1, "\n")

	model.eval()
	sigmoid = nn.Sigmoid()
	seq, mask, labels = next(iter(testloader))

	output = model(seq.to(device), mask.to(device))
	output = sigmoid(output)
	prediction = (output > threshold).int().cpu()

	accuracy = accuracy_score(labels, prediction)

	print("Subset Accuracy:", accuracy)
	print(classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=False))
	print("Best Dev F1:", bestDevF1, "at epoch", bestEpochDevF1, "\n")
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
	multilabel_confusion_matrix(np.array(labels), output.cpu().detach().numpy(), emotions, top_x=3, filename=filename)


if __name__ == "__main__":
	main()