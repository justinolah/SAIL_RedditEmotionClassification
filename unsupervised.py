from learn import *
from helpers import *
from wordnet import getDefinition
from nltk.corpus import stopwords
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
	centroids = torch.Tensor(len(emotions), 768)
	for i, emotion in enumerate(emotions):
		centroid = vecs[labels[:,i] == 1].mean(axis=0)
		centroids[i,:] = centroid
	return centroids


def getWordRep(texts, wordEmbedding, stop_words, dim):
	vecs = torch.Tensor(len(texts), dim)
	for i, text in tqdm(enumerate(texts), total=len(texts)):
		text = text.lower()
		text = re.sub(r"[^a-z\s]+", " ", text)
		text = re.sub(r"\s+", " ", text)
		words = text.split()

		words = [word for word in words if word not in stop_words]
		
		embeds = [wordEmbedding[word] for word in words if torch.count_nonzero(wordEmbedding[word]) > 0]

		if len(embeds) == 0:
			vecs[i,:] = torch.zeros(dim)
		else:
			vecs[i,:] = torch.mean(torch.stack(embeds), 0)
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
		print(f"{emotion}: {best} (F1: {f1s[best_index]}, support: {np.sum(targets[:,i])})")
		thresholds.append(best)

	thresholds = np.array(thresholds)
	return thresholds

def main():
	if torch.cuda.is_available():
		print("Using cuda...")
		torch.cuda.set_device(2)
		device = torch.device("cuda:2")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	max_length = 128
	batch_size = 16
	framework = "Unsupervised with Goemotions trained bert embeddings"
	grouping = None
	dataset = "semeval"
	goemotions_trained = True
	defintion = True
	dim = 200

	config.framework = framework
	config.grouping = grouping
	config.dataset = dataset
	config.goemotions_trained = goemotions_trained
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

	print(f"Dev set: {len(dev)}")
	print(f"Test set: {len(all_data)}")

	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	if dataset == "semeval":
		test_set = makeBERTDatasetSemEval(all_data, tokenizer, max_length, newEmotions)
		dev_set = makeBERTDatasetSemEval(dev, tokenizer, max_length, newEmotions)
	elif dataset == "goemotions":
		test_set = makeBERTDatasetGoEmotions(all_data, tokenizer, max_length, newEmotions)
		dev_set = makeBERTDatasetGoEmotions(dev, tokenizer, max_length, newEmotions)

	testloader = DataLoader(test_set, batch_size=batch_size)
	devloader = DataLoader(dev_set, batch_size=batch_size)

	bert = BertModel.from_pretrained('bert-base-uncased')

	model = BERT_Model(bert, len(emotions))
	model = model.to(device)
	sigmoid = nn.Sigmoid()

	if goemotions_trained == True:
		checkpoint = torch.load(bertfile, map_location=device)
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
	stop_words = stopwords.words('english')
	emotion_word_vecs = getWordRep(newEmotions, wordEmbedding, stop_words, dim) #todo use mean of synoyms
	if dataset == "semeval":
		word_vecs_dev = getWordRep(dev.Tweet.tolist(), wordEmbedding, stop_words, dim)
		word_vecs_test = getWordRep(all_data.Tweet.tolist(), wordEmbedding, stop_words, dim)
	elif dataset == "goemotions":
		word_vecs_dev = getWordRep(dev.text.tolist(), wordEmbedding, stop_words, dim)
		word_vecs_test = getWordRep(all_data.test.tolist(), wordEmbedding, stop_words, dim)

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
	print("Word Thresholds:")
	thresholds_word = tuneThresholds(word_similarities, dev_targets, newEmotions, np.linspace(0 ,0.8, num=40))

	#Evaluation
	sentence_vecs, targets = getSentenceRep(testloader, model, device)

	if dataset == "semeval":
		texts = all_data.Tweet.tolist()
	elif dataset == "goemotions":
		texts = all_data.text.tolist()

	predictions = []
	predictions_centroids = []
	predictions_word = []

	for i, (vec, word_vec) in enumerate(zip(sentence_vecs, word_vecs_test)):
		similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
		centroid_similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), centroids.to(device))
		word_similarities = F.cosine_similarity(word_vec.unsqueeze(0).to(device), emotion_word_vecs.to(device))

		pred = (similarities.detach().cpu().numpy() > thresholds).astype(int)
		pred_centroid = (centroid_similarities.detach().cpu().numpy() > thresholds_centroids).astype(int)
		pred_word = (word_similarities.detach().cpu().numpy() > thresholds_word).astype(int)

		if i < 5:
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred.tolist()) if num == 1])}")
			print("Sentence Similarity")
			for index in similarities.argsort(descending=True):
				print(f"label: {newEmotions[index]}, similarity: {similarities[index]}") 
			print("")
		elif i < 10:
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred_centroid.tolist()) if num == 1])}")
			print("Centroid Similarity")
			for index in centroid_similarities.argsort(descending=True):
				print(f"label: {newEmotions[index]}, similarity: {centroid_similarities[index]}") 
			print("")
		elif i < 15:
			print(texts[i])
			print(f"actual label: {','.join([newEmotions[index] for index, num in enumerate(targets[i].tolist()) if num == 1])}") 
			print(f"predicted label: {','.join([newEmotions[index] for index, num in enumerate(pred_word.tolist()) if num == 1])}")
			print("Word Similarity")
			for index in word_similarities.argsort(descending=True):
				print(f"label: {newEmotions[index]}, similarity: {word_similarities[index]}") 
			print("")

		predictions.append(pred)
		predictions_centroids.append(pred_centroid)
		predictions_word.append(pred_word)
			
	print("Sentence Similarity:")
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