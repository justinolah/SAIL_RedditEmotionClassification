import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel, RobertaTokenizerFast
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score

from tqdm import tqdm

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

class RoBERTa_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(RoBERTa_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
		out = self.fc(cls_hs)
		return out


def makeBERTDataset(data, tokenizer, max_length, emotions, new_emotions=None, idx_map=None):
	if idx_map is None:
		data.labels = data.labels.apply(lambda x: getYMatrix(x,len(emotions)))
	else:
		data.labels = data.labels.apply(lambda x: getYMatrixWithMap(x, len(new_emotions), idx_map))

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
	sigmoid = nn.Sigmoid()

	model.train()

	train_running_loss = 0.0
	traintargets = []
	trainpredictions = []

	loop = tqdm(enumerate(trainloader), total=len(trainloader), leave=False)
	for i, batch in loop:
		seq, mask, labels = batch
		traintargets.append(labels.detach())

		optimizer.zero_grad()
		output = model(seq.to(device), mask.to(device))

		loss = loss_fn(output, labels.float().to(device))
		loss.backward()
		optimizer.step()

		output = sigmoid(output)
		trainpredictions.append((output.cpu() > threshold).int().detach())

		train_running_loss += loss.item()

		loop.set_postfix(loss = loss.item())

	trainLoss = train_running_loss / len(trainloader)

	trainpredictions = np.concatenate(trainpredictions)
	traintargets = np.concatenate(traintargets)
	f1_train = f1_score(traintargets, trainpredictions, average='macro')

	#dev evaluation
	model.eval()
	dev_running_loss = 0.0
	devtargets = []
	devpredictions = []
	for batch in devloader:
		seq, mask, labels = batch
		devOutput = model(seq.to(device), mask.to(device))

		dev_running_loss += loss_fn(devOutput, labels.float().to(device)).item()

		devOutput = sigmoid(devOutput)
		devpredictions.append((devOutput.cpu() > threshold).int().detach())
		devtargets.append(labels)

	devLoss = dev_running_loss / len(devloader)

	devpredictions = np.concatenate(devpredictions)
	devtargets = np.concatenate(devtargets)
	f1_dev = f1_score(devtargets, devpredictions, average='macro')

	return trainLoss, devLoss, f1_train, f1_dev

def main():
	epochs = 5
	batch_size = 16
	max_length = 128
	weight_decay = 0
	threshold = 0.5
	lr = 5e-5
	init_lr = lr
	filename = "roberta"
	framework = "roberta"
	grouping = None

	config.epochs = epochs
	config.batch_size = batch_size
	config.max_length = max_length
	config.weight_decay = weight_decay
	config.lr = lr
	config.framework = framework
	config.grouping = grouping

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	#make Datasets
	emotions = getEmotions()
	emotions.remove("neutral")

	ekmanDict = getEkmanDict()
	sentDict = getSentimentDict()
	ekEmotions = ekmanDict.keys()
	sentEmotions = sentDict.keys()
	ek_idx_map = getEmotionIndexMap(emotions, ekmanDict)
	sent_idx_map = getEmotionIndexMap(emotions, sentDict)

	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base')

	if grouping == "ekman":
		new_emotions = ekEmotions
		idx_map = ek_idx_map
	elif grouping == "sentiment":
		new_emotions = sentEmotions
		idx_map = sent_idx_map
	else:
		new_emotions = emotions
		idx_map = None

	train_set = makeBERTDataset(train, tokenizer, max_length, emotions, new_emotions=new_emotions, idx_map=idx_map)
	test_set = makeBERTDataset(test, tokenizer, max_length, emotions, new_emotions=new_emotions, idx_map=idx_map)
	dev_set = makeBERTDataset(dev, tokenizer, max_length, emotions, new_emotions=new_emotions, idx_map=idx_map)

	trainloader = DataLoader(train_set, batch_size=batch_size)
	testloader = DataLoader(test_set, batch_size=batch_size)
	devLoader = DataLoader(dev_set, batch_size=batch_size)

	if grouping is None:
		pos_weight = torch.sqrt(torch.div((len(train.labels) - torch.sum(torch.tensor(train.labels),0)), torch.sum(torch.tensor(train.labels),0))).to(device)
	else:
		pos_weight = None
	
	if grouping == "ekman":
		emotions = ekEmotions
	elif grouping == "sentiment":
		emotions = sentEmotions

	#initialize model
	bert = RobertaModel.from_pretrained('roberta-base')

	model = RoBERTa_Model(bert, len(emotions))
	model = model.to(device)
	wandb.watch(model)

	loss_fn= nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

		wandb.log({"train_loss": epoch_loss,
			"dev_loss": dev_loss,
			"train_f1": f1_train,
			"dev_f1": f1_dev
		})

		if epoch == 0 or np.argmax(devF1) == epoch:
			print("saving checkpoint...")
			torch.save({
				'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'devF1' : devF1[-1],
				}, "roberta.pt")

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
	bestCheckpoint = torch.load("roberta.pt")
	model.load_state_dict(bestCheckpoint['model_state_dict'])
	bestEpochDevF1 = bestCheckpoint['epoch']
	bestDevF1 = bestCheckpoint['devF1']
	print("Best Dev F1:", bestDevF1, "at epoch", bestEpochDevF1, "\n")

	model.eval()
	sigmoid = nn.Sigmoid()
	targets = []
	outputs = []
	predictions = []
	for batch in testloader:
		seq, mask, labels = batch
		output = model(seq.to(device), mask.to(device))

		output = sigmoid(output)
		output = output.cpu()
		outputs.append(output.detach())
		predictions.append((output > threshold).int().detach())
		targets.append(labels)

	predictions = np.concatenate(predictions)
	targets = np.concatenate(targets)
	outputs = np.concatenate(outputs)

	accuracy = accuracy_score(targets, predictions)

	print("Subset Accuracy:", accuracy)
	print(classification_report(targets, predictions, target_names=emotions, zero_division=0, output_dict=False))
	print("Best Dev F1:", bestDevF1, "at epoch", bestEpochDevF1, "\n")
	report = classification_report(targets, predictions, target_names=emotions, zero_division=0, output_dict=True)

	table = wandb.Table(dataframe=pd.DataFrame.from_dict(report))
	wandb.log({"report": table})

	#export resuls to csv
	micro = list(report['micro avg'].values())
	micro.pop() 
	macro = list(report['macro avg'].values())
	macro.pop()
	scores = [accuracy, *micro, *macro]
	results = pd.DataFrame(data=[scores], columns=['accuracy', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'])
	results.to_csv("tables/" + filename + "_results.csv")

	#confusion matrix
	multilabel_confusion_matrix(np.array(targets), np.array(outputs), emotions, top_x=3, filename=filename)


if __name__ == "__main__":
	main()