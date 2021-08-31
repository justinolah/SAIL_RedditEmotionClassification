from learn import *
from helpers import *
from wordnet import getDefinition
from transformers import BertModel, BertTokenizerFast
from torchtext.vocab import GloVe

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

def getCentroids(vecs, labels, emotions):
	centroids = []
	for i, emotion in enumerate(emotions):
		centroid = vecs[labels[:,i] == 1].mean(axis=0)
		centroids.append(centroid)
	return centroid


def getWordRep(texts, wordEmbedding, stopwords, dim):
	vecs = []
	for text in tqdm(texts):
		text = text.lower()
		text = re.sub(r"[^a-z\s]+", " ", text)
		text = re.sub(r"\s+", " ", text)
		words = text.split()

		words = [word for word in words if word not in stopwords]
		
		embeds = [wordEmbedding[word].numpy() for word in words if torch.count_nonzero(wordEmbedding[word]) > 0]

		if len(embeds) == 0:
			vecs.append(np.zeros(dim))
		else:
			vecs.append(np.array(embeds).mean(axis=0))
	return vecs

def tuneThresholds(similarities, targets, emotions, threshold_options):
	thresholds = []
	
	for i, emotion in enumerate(emotions):
		f1s = []
		for threshold in threshold_options:
			predictions = []
			for sim in similarities:
				predictions.append(int(sim[i] > threshold))
			f1s.append(f1_score(targets[:,i], predictions))

		best_index = np.argmax(f1s)
		best = threshold_options[best_index]
		print(f"{emotion}: {best} (F1: {f1s[best_index]}, support: {np.sum(dev_targets[:,i])})")
		thresholds.append(best)
	
	"""
	for i, emotion in enumerate(newEmotions):
		threshold = np.mean(similarities[dev_targets[:,i] == 1,i])
		print(f"{emotion}: {threshold}")
		thresholds.append(threshold)
	"""

	thresholds = np.array(thresholds)
	return thresholds

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		torch.cuda.set_device(3)
		device = torch.device("cuda:3")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	max_length = 128
	batch_size = 16
	framework = "Unsupervised with Goemotions trained bert embeddings"
	grouping = None
	dataset = "semeval"
	defintion = True
	dim = 200

	config.framework = framework
	config.grouping = grouping
	config.dataset = dataset
	config.defintion = defintion

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
		all_data = pd.concat([train, test])
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))
		dev.Tweet = dev.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
		dev.Tweet = dev.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))
	elif dataset == "goemotions":
		newEmotions = getEmotions()
		newEmotions.remove("neutral")
		#train = getTrainSet()
		test = getTestSet()
		dev = getValSet()
		all_data = test
	else:
		print("Invalid dataset")
		return

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

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

	checkpoint = torch.load(bertfile)
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()

	#expand labels with definitions
	expanded = []
	for emotion in newEmotions:
		expanded.append(f"{emotion}: {getDefinition(emotion)}")

	emotion_input = tokenizer.batch_encode_plus(
		(expanded if defintion == True else newEmotions),
		max_length = max_length,
		padding='max_length',
		truncation=True
	)

	emotion_ids = torch.tensor(emotion_input['input_ids'])
	emotion_mask = torch.tensor(emotion_input['attention_mask'])

	emotion_vecs = model(emotion_ids.to(device), emotion_mask.to(device))
	emotion_vecs = emotion_vecs.cpu()

	#Glove word embeddings
	wordEmbedding = GloVe(name='twitter.27B', dim=dim)
	stopwords = getStopWords()
	emotion_word_vecs = np.array([wordEmbedding[emotion].numpy() for emotion in newEmotions]) #todo use mean of synoyms
	if dataset == "semeval":
		word_vecs_dev = getWordRep(dev.Tweet.tolist(), wordEmbedding, stopwords, dim)
		word_vecs_test = getWordRep(all_data.Tweet.tolist(), wordEmbedding, stopwords, dim)
	elif dataset == "goemotions":
		word_vecs_dev = getWordRep(dev.text.tolist(), wordEmbedding, stopwords, dim)
		word_vecs_test = getWordRep(all_data.test.tolist(), wordEmbedding, stopwords, dim)

	#dev tunings
	dev_vectors, dev_targets = getSentenceRep(devloader, model, device)
	centroids = getCentroids(dev_vectors, dev_targets, newEmotions)
	similarities = []
	centroid_similarities = []
	for i, vec in enumerate(dev_vectors):
		sim = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		centroid_sim = F.cosine_similarity(vec.unsqueeze(0).to(device), centroids.to(device))
		#sim = sigmoid(sim)
		similarities.append(sim)
		centroid_similarities.append(centroid_sim)

	word_similarities = []
	for i, vec in enumerate(word_vecs_dev):
		sim = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_word_vecs.to(device))
		word_similarities.append(sim)
	
	threshold_options = np.linspace(0.4,0.95, num=30)
	print("Sentence Rep Thresholds:")
	thresholds = tuneThresholds(similarities, dev_targets, newEmotions, threshold_options)
	print("Centroid Thresholds:")
	thresholds_centroids = tuneThresholds(centroid_similarities, dev_targets, newEmotions, threshold_options)
	print("Word Centroids:")
	thresholds_word = tuneThresholds(word_similarities, dev_targets, newEmotions, threshold_options)

	#Evaluation
	vectors, targets = getSentenceRep(dataloader, model, device)

	if dataset == "semeval":
		texts = all_data.Tweet.tolist()
	elif dataset == "goemotions":
		texts = all_data.text.tolist()

	predictions = []
	predictions_centroids = []
	predictions_word = []

	for i, vec in enumerate(vectors):
		similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		#similarities = sigmoid(similarities)

		centroid_similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), centroids.to(device))

		closest = similarities.argsort(descending=True)

		pred = (similarities.detach().cpu().numpy() > thresholds).astype(int)
		pred_centroid = (centroid_similarities.detach().cpu().numpy() > thresholds).astype(int)

		if i < 5:
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred.tolist()) if num == 1])}")
			for index in closest:
				print(f"label: {newEmotions[index]}, similarity: {similarities[index]}") 
			print("")
		elif i < 20:
			index = closest[0]
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}")
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred.tolist()) if num == 1])}")
			print(f"label: {newEmotions[index]}, similarity: {similarities[index]}\n")

		#pred = np.zeros(len(newEmotions))
		#pred[closest[0]] = 1
		predictions.append(pred)
		predictions_centroids.append(pred_centroid)

	for i, vec in enumerate(word_vecs_test):
		word_similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_word_vecs.to(device))
		pred = (word_similarities.detach().cpu().numpy() > thresholds_word).astype(int)
		predictions_word.append(pred)
		
	print("Setence Similarity:")
	print(classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=False))
	report = classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=True)
	print("")
	print("Word Similarity:")
	print(classification_report(targets, predictions_word, target_names=newEmotions, zero_division=0, output_dict=False))
	print("")
	print("Centroid Similarity:")
	print(classification_report(targets, predictions_centroids, target_names=newEmotions, zero_division=0, output_dict=False))
	print("")

	table = wandb.Table(dataframe=pd.DataFrame.from_dict(report))
	wandb.log({"Unsupervised": table})





if __name__ == '__main__':
	main()