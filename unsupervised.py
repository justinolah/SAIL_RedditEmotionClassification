from learn import *
from helpers import *
from transformers import BertModel, BertTokenizerFast

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score

from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

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

DIR = "SemEval2018-Task1-all-data/English/E-c/"

TRAIN_DIR = "2018-E-c-En-train.txt"
TEST_DIR = "2018-E-c-En-test-gold.txt"
DEV_DIR = "2018-E-c-En-dev.txt"

SEMEVAL_EMOTIONS_FILE = "data/SemEvalEmotions.txt"

def makeBERTDatasetSemEval(data, tokenizer, max_length, emotions):
	tokens = tokenizer.batch_encode_plus(
		data.Tweet.tolist(),
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	seq = torch.tensor(tokens['input_ids'])
	mask = torch.tensor(tokens['attention_mask'])
	y = torch.tensor(data[emotions].values)

	dataset = TensorDataset(seq, mask, y)
	return dataset

def makeBERTDatasetGoEmotions(data, tokenizer, max_length, emotions):
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

class BERT_Model(nn.Module):
	def __init__(self, bert, numEmotions):
		super(BERT_Model, self).__init__()
		self.bert = bert 
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		_, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
		#out = self.fc(cls_hs)
		return cls_hs

def getSemEvalEmotions():
	with open(SEMEVAL_EMOTIONS_FILE) as f:
		return f.read().splitlines()

def getSentenceRep(dataloader, model, device):
	outputs = []	
	targets = []
	num = 0

	for batch in tqdm(dataloader):
		seq, mask, labels = batch
		num += len(seq)

		targets.append(labels.detach())

		output = model(seq.to(device), mask.to(device))
		outputs.append(output.detach().cpu())

	vectors = torch.Tensor(num, 768)
	torch.cat(outputs, out=vectors)
	targets = np.concatenate(targets)

	return vectors, targets

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	max_length = 128
	batch_size = 16
	framework = "Unsupervised with Goemotions trained bert embeddings"
	grouping = None
	dataset = "semeval"

	config.framework = framework
	config.grouping = grouping
	config.dataset = dataset

	if grouping == "sentiment":
		emotions = getSentimentDict().keys()
		bertfile = "bert_sentiment.pt"
	else:
		emotions = getEmotions()
		emotions.remove("neutral")
		bertfile = "bert_best.pt"

	if dataset == "semeval":
		newEmotions = getSemEvalEmotions()
		train = pd.read_csv(DIR + TRAIN_DIR, sep='\t')
		test = pd.read_csv(DIR + TEST_DIR, sep='\t')
		dev = pd.read_csv(DIR + DEV_DIR, sep='\t')
	elif dataset == "goemotions":
		newEmotions = getEmotions()
		newEmotions.remove("neutral")
		train = getTrainSet()
		test = getTestSet()
		dev = getValSet()
	else:
		print("Invalid dataset")
		return

	#todo expand emotion labels with wordnet synonyms, defintion, etc.

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	all_data = pd.concat([train, test])

	if dataset == "semeval":
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))
		dev.Tweet = dev.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
		dev.Tweet = dev.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))

	print(f"Number of tweets: {len(all_data)}")

	if dataset == "semeval":
		test_set = makeBERTDatasetSemEval(all_data, tokenizer, max_length, newEmotions)
		dev_set = makeBERTDatasetSemEval(dev, tokenizer, max_length, newEmotions)
	elif dataset == "goemotions":
		test_set = makeBERTDatasetGoEmotions(all_data, tokenizer, max_length, newEmotions)
		dev_set = makeBERTDatasetGoEmotions(dev, tokenizer, max_length, newEmotions)

	dataloader = DataLoader(test_set, batch_size=batch_size)
	devloader = DataLoader(dev_set, batch_size=batch_size)

	bert = BertModel.from_pretrained('bert-base-uncased')

	model = BERT_Model(bert, len(emotions))
	model = model.to(device)
	sigmoid = nn.Sigmoid()

	#checkpoint = torch.load(bertfile)
	#model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	emotion_input = tokenizer.batch_encode_plus(
		newEmotions,
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	emotion_ids = torch.tensor(emotion_input['input_ids'])
	emotion_mask = torch.tensor(emotion_input['attention_mask'])

	emotion_vecs = model(emotion_ids.to(device), emotion_mask.to(device))
	emotion_vecs = emotion_vecs.cpu()

	#dev tunings
	dev_vectors, dev_targets = getSentenceRep(devloader, model, device)
	similarities = []
	for i, vec in enumerate(dev_vectors):
		sim = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		sim = sigmoid(sim)
		similarities.append(sim)

	threshold_options = np.linspace(0,1, num=100)
	thresholds = []
	print("Thresholds:")
	for i, emotion in enumerate(newEmotions):
		f1s = []
		for threshold in threshold_options:
			predictions = []
			for sim in similarities:
				predictions.append(int(sim[i] > threshold))
			f1s.append(f1_score(dev_targets[:,i], predictions))

		best_index = np.argmax(f1s)
		best = threshold_options[best_index]
		print(f"{emotion}: {best} (F1: {f1s[best_index]})")
		thresholds.append(best)

	thresholds = np.array(thresholds)

	#Evaluation
	vectors, targets = getSentenceRep(dataloader, model, device)
	predictions = []

	if dataset == "semeval":
		texts = all_data.Tweet.tolist()
	elif dataset == "goemotions":
		texts = all_data.text.tolist()

	for i, vec in enumerate(vectors):
		similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		similarities = sigmoid(similarities)
		closest = similarities.argsort(descending=True)

		pred = (similarities > threshold).int().detach().cpu()

		if i < 5:
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred.tolist()) if num == 1])}")
			for index in closest:
				print(f"label: {newEmotions[index]}, similarity: {similarities[index]}\n") 
		elif i < 20:
			index = closest[0]
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}")
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred.tolist()) if num == 1])}")
			print(f"label: {newEmotions[index]}, similarity: {similarities[index]}\n")

		#pred = np.zeros(len(newEmotions))
		#pred[closest[0]] = 1
		predictions.append(pred)

	print(classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=False))
	report = classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=True)

	table = wandb.Table(dataframe=pd.DataFrame.from_dict(report))
	wandb.log({"Unsupervised": table})





if __name__ == '__main__':
	main()