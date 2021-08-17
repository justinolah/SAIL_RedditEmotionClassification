import pandas as pd
from helpers import *
from transformers import BertModel, BertTokenizerFast

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

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

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	max_length = 128
	batch_size = 16
	grouping = None

	if grouping == "sentiment":
		emotions = getSentimentDict().keys()
		bertfile = "bert_sentiment.pt"
	else:
		emotions = getEmotions()
		emotions.remove("neutral")
		bertfile = "bert_best.pt"

	semEmotions = getSemEvalEmotions()

	#todo expand emotion labels with wordnet synonyms, defintion, etc.

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	train = pd.read_csv(DIR + TRAIN_DIR, sep='\t')
	test = pd.read_csv(DIR + TEST_DIR, sep='\t')
	dev = pd.read_csv(DIR + DEV_DIR, sep='\t')

	all_data = pd.concat([train, test, dev])
	all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
	all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))

	print(f"Number of tweets: {len(all_data)}")

	data_set = makeBERTDatasetSemEval(all_data, tokenizer, max_length, semEmotions)
	dataloader = DataLoader(data_set, batch_size=batch_size)

	bert = BertModel.from_pretrained('bert-base-uncased')

	model = BERT_Model(bert, len(emotions))
	model = model.to(device)

	checkpoint = torch.load(bertfile)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	emotion_input = tokenizer.batch_encode_plus(
		semEmotions,
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	emotion_ids = torch.tensor(emotion_input['input_ids'])
	emotion_mask = torch.tensor(emotion_input['attention_mask'])

	emotion_vecs = model(emotion_ids.to(device), emotion_mask.to(device))
	emotion_vecs = emotion_vecs.cpu()

	outputs = []	
	targets = []

	for batch in tqdm(dataloader):
		seq, mask, labels = batch

		targets.append(labels.detach())

		output = model(seq.to(device), mask.to(device))
		outputs.append(output.detach().cpu())

	vectors = torch.Tensor(len(dataloader), 768)
	torch.cat(outputs, out=vectors)

	targets = np.concatenate(targets)
	predictions = []

	tweets = all_data.Tweet.tolist()

	for i, vec in enumerate(vectors):
		similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		closest = similarities.argsort(descending=True)
		if i < 5:
			print(tweets[i])
			for index in closest:
				print(f"actual label: {','.join([semEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
				print(f"label: {semEmotions[index]}, similarity: {similarities[index]}\n") 
		elif i < 20:
			index = closest[0]
			print(tweets[i])
			print(f"actual label: {','.join([semEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}")
			print(f"label: {semEmotions[index]}, similarity: {similarities[index]}\n")

		pred = np.zeros(len(semEmotions))
		pred[closest[0]] = 1
		predictions.append(pred)

	print(classification_report(targets, predictions, target_names=semEmotions, zero_division=0, output_dict=False))





if __name__ == '__main__':
	main()