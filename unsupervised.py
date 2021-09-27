from learn import *
from helpers import *
from wordnet import getDefinition

import nltk
from nltk.corpus import stopwords
from transformers import BertModel, BertTokenizerFast, AutoTokenizer, AutoModel
from torchtext.vocab import GloVe

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

from tqdm import tqdm

import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F

wandb.init(project='SAILGoemotions', entity='justinolah')
config = wandb.config

seed_value = 42
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

class SBERT_Model(nn.Module):
	def __init__(self, sbert, numEmotions):
		super(SBERT_Model, self).__init__()
		self.sbert = sbert
		self.fc = nn.Linear(768, numEmotions)

	def forward(self, sent_id, mask):
		output = self.sbert(sent_id, attention_mask=mask)
		return self.mean_pooling(output, mask)

	def mean_pooling(self, model_output, attention_mask):
		token_embeddings = model_output[0] #First element of model_output contains all token embeddings
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
		sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
		sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
		return sum_embeddings / sum_mask

def getSemEvalEmotions():
	with open(SEMEVAL_EMOTIONS_FILE) as f:
		return f.read().splitlines()

def getSentenceRep(dataloader, model, sentence_dim, device):
	outputs = []	
	targets = []
	num = 0

	for batch in tqdm(dataloader):
		seq, mask, labels = batch
		num += len(seq)

		targets.append(labels.detach())

		output = model(seq.to(device), mask.to(device))
		outputs.append(output.detach().cpu())

	vectors = torch.Tensor(num, sentence_dim)
	torch.cat(outputs, out=vectors)
	targets = np.concatenate(targets)

	return vectors, targets

def getCentroids(vecs, labels, emotions, sentence_dim):
	centroids = torch.Tensor(len(emotions), sentence_dim)
	for i, emotion in enumerate(emotions):
		v = vecs[labels[:,i] == 1]
		if len(v) > 0:
			centroid = v.mean(axis=0)
			centroids[i,:] = centroid
		else:
			centroids[i,:] = torch.zeros(sentence_dim)
	return centroids


def getWordRep(texts, wordEmbedding, stop_words, word_dim):
	vecs = torch.Tensor(len(texts), word_dim)
	for i, text in tqdm(enumerate(texts), total=len(texts)):
		text = text.lower()
		text = re.sub(r"[^a-z\s]+", " ", text)
		text = re.sub(r"\s+", " ", text)
		words = text.split()

		words = [word for word in words if word not in stop_words]
		
		embeds = [wordEmbedding[word] for word in words if torch.count_nonzero(wordEmbedding[word]) > 0]

		if len(embeds) == 0:
			vecs[i,:] = torch.zeros(word_dim)
		else:
			vecs[i,:] = torch.mean(torch.stack(embeds), 0)
	return vecs

def tuneThresholds(similarities, targets, emotions, threshold_options):
	if len(similarities) == 0:
		return 0.5 * np.ones(len(emotions))

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
	print("")

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

	confusion = True
	save_report = True
	print_examples = True

	max_length = 128
	batch_size = 16
	framework = "Unsupervised with Goemotions trained bert embeddings"
	grouping = None
	dataset = "semeval"
	tune_thresholds = True
	goemotions_trained = True
	defintion = True
	word_dim = 200
	sentence = "s-bert"

	config.framework = framework
	config.grouping = grouping
	config.dataset = dataset
	config.sentence = sentence
	config.tune_thresholds = tune_thresholds
	config.goemotions_trained = goemotions_trained
	config.defintion = defintion

	if grouping == "sentiment":
		emotions = getSentimentDict().keys()
		bertfile = "bert_sentiment.pt"
	else:
		emotions = getEmotions()
		emotions.remove("neutral")
		if sentence == "s-bert":
			bertfile = "sbert_large.pt"
		else:
			bertfile = "bert_best.pt"

	if dataset == "semeval":
		newEmotions = getSemEvalEmotions()
		top_x = 2
		train = pd.read_csv(DIR + TRAIN_DIR, sep='\t')
		test = pd.read_csv(DIR + TEST_DIR, sep='\t')
		dev = pd.read_csv(DIR + DEV_DIR, sep='\t')
		all_data = pd.concat([train, test, dev])
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"\B@\w+", "@mention", x))
		all_data.Tweet = all_data.Tweet.apply(lambda x: re.sub(r"&amp;", "&", x))

		splits = list(np.linspace(0.995, 0.94, num=8))
	elif dataset == "goemotions":
		top_x = 3
		newEmotions = getEmotions()
		newEmotions.remove("neutral")
		test = getTestSet()
		dev = getValSet()
		all_data = pd.concat([test, dev])

		splits = list(np.linspace(0.995, 0.90, num=8))
	else:
		print("Invalid dataset")
		return

	if sentence == "s-bert":
		sbert = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
		tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
		sentence_dim = 768
		model = SBERT_Model(sbert, len(emotions))
		model = model.to(device)
	else:
		bert = BertModel.from_pretrained('bert-base-uncased')
		tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
		sentence_dim = 768
		model = BERT_Model(bert, len(emotions))
		model = model.to(device)

	if goemotions_trained == True:
		checkpoint = torch.load(bertfile, map_location=device)
		model.load_state_dict(checkpoint['model_state_dict'])

	config.sentence_dim = sentence_dim
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
	emotion_vecs = emotion_vecs

	wordEmbedding = GloVe(name='twitter.27B', dim=word_dim)
	stop_words = stopwords.words('english')
	emotion_word_vecs = getWordRep(newEmotions, wordEmbedding, stop_words, word_dim)

	sentence_precision, sentence_recall, sentence_f1 = [], [], []
	word_precision, word_recall, word_f1 = [], [], []
	centroid_precision, centroid_recall, centroid_f1 = [], [], []

	if dataset == "semeval":
		data_set = makeBERTDatasetSemEval(all_data, tokenizer, max_length, newEmotions)
	elif dataset == "goemotions":
		data_set = makeBERTDatasetGoEmotions(all_data, tokenizer, max_length, newEmotions)

	loader = DataLoader(data_set, batch_size=batch_size)

	#Glove word embeddings
	if dataset == "semeval":
		word_vecs_all = getWordRep(all_data.Tweet.tolist(), wordEmbedding, stop_words, word_dim)
	elif dataset == "goemotions":
		word_vecs_all = getWordRep(all_data.text.tolist(), wordEmbedding, stop_words, word_dim)

	sentence_vectors_all, targets_all = getSentenceRep(loader, model, sentence_dim, device)

	dev_set_size = []
	for testsplit in splits:
		print("************************************************")
		print(f"Test Split: {testsplit}")

		dev_indices, test_indices = train_test_split([i for i in range(len(all_data))], test_size=testsplit)

		print(f"Dev Set: {len(dev_indices)}")
		print(f"Test Set: {len(test_indices)}")
		dev_set_size.append(len(dev_indices))

		word_vecs_dev = word_vecs_all[dev_indices]
		word_vecs_test = word_vecs_all[test_indices]

		dev_vectors = sentence_vectors_all[dev_indices]
		dev_targets = targets_all[dev_indices]
		centroids = getCentroids(dev_vectors, dev_targets, newEmotions, sentence_dim)

		sentence_vecs = sentence_vectors_all[test_indices]
		targets = targets_all[test_indices]

		#dev tunings
		similarities = []
		centroid_similarities = []
		for i, vec in enumerate(dev_vectors):
			sim = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
			centroid_sim = F.cosine_similarity(vec.unsqueeze(0).to(device), centroids.to(device))
			similarities.append(sim)
			centroid_similarities.append(centroid_sim)

		word_similarities = []
		for i, vec in enumerate(word_vecs_dev):
			sim = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_word_vecs.to(device))
			word_similarities.append(sim)
		
		if tune_thresholds == True:
			threshold_options = np.linspace(0,1, num=30)
			print("Sentence Rep Thresholds:")
			thresholds = tuneThresholds(similarities, dev_targets, newEmotions, threshold_options)
			print("Centroid Thresholds:")
			thresholds_centroids = tuneThresholds(centroid_similarities, dev_targets, newEmotions, threshold_options)
			print("Word Thresholds:")
			thresholds_word = tuneThresholds(word_similarities, dev_targets, newEmotions, threshold_options)
		else:
			thresholds = 0.5 * np.ones(len(newEmotions))
			thresholds_centroids = 0.5 * np.ones(len(newEmotions))
			thresholds_word = 0.5 * np.ones(len(newEmotions))

		#Evaluation
		if dataset == "semeval":
			texts = all_data.Tweet.to_numpy()[test_indices]
		elif dataset == "goemotions":
			texts = all_data.text.to_numpy()[test_indices]

		predictions = []
		predictions_centroids = []
		predictions_word = []

		outputs_sentence = []
		outputs_centroid = []
		outputs_word = []

		for i, (vec, word_vec) in enumerate(zip(sentence_vecs, word_vecs_test)):
			similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), emotion_vecs.to(device))
			centroid_similarities = F.cosine_similarity(vec.unsqueeze(0).to(device), centroids.to(device))
			word_similarities = F.cosine_similarity(word_vec.unsqueeze(0).to(device), emotion_word_vecs.to(device))

			pred = (similarities.detach().cpu().numpy() > thresholds).astype(int)
			pred_centroid = (centroid_similarities.detach().cpu().numpy() > thresholds_centroids).astype(int)
			pred_word = (word_similarities.detach().cpu().numpy() > thresholds_word).astype(int)

			outputs_sentence.append(similarities.detach().cpu())
			outputs_centroid.append(centroid_similarities.detach().cpu())
			outputs_word.append(word_similarities.detach().cpu())

			predictions.append(pred)
			predictions_centroids.append(pred_centroid)
			predictions_word.append(pred_word)

			if print_examples == True:
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
				print_examples = False

		outputs_sentence = np.stack(outputs_sentence)
		outputs_centroid = np.stack(outputs_centroid)
		outputs_word = np.stack(outputs_word)
				
		print("Sentence Similarity:")
		print(classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=False), "\n")
		print("Word Similarity:")
		print(classification_report(targets, predictions_word, target_names=newEmotions, zero_division=0, output_dict=False), "\n")
		print("Centroid Similarity:")
		print(classification_report(targets, predictions_centroids, target_names=newEmotions, zero_division=0, output_dict=False), "\n")

		report_sentence = classification_report(targets, predictions, target_names=newEmotions, zero_division=0, output_dict=True)
		report_word = classification_report(targets, predictions_word, target_names=newEmotions, zero_division=0, output_dict=True)
		report_centroid = classification_report(targets, predictions_centroids, target_names=newEmotions, zero_division=0, output_dict=True)

		sentence_precision.append(report_sentence['macro avg']['precision'])
		sentence_recall.append(report_sentence['macro avg']['recall'])
		sentence_f1.append(report_sentence['macro avg']['f1-score'])

		word_precision.append(report_word['macro avg']['precision'])
		word_recall.append(report_word['macro avg']['recall'])
		word_f1.append(report_word['macro avg']['f1-score'])

		centroid_precision.append(report_centroid['macro avg']['precision'])
		centroid_recall.append(report_centroid['macro avg']['recall'])
		centroid_f1.append(report_centroid['macro avg']['f1-score'])


		if save_report == True:
			table_sentence = wandb.Table(dataframe=pd.DataFrame.from_dict(report_sentence))
			table_word = wandb.Table(dataframe=pd.DataFrame.from_dict(report_word))
			table_centroid = wandb.Table(dataframe=pd.DataFrame.from_dict(report_centroid))
			wandb.log({"Sentence Rep": table_sentence, "Word Rep": table_word, "Centroid Rep": table_centroid})
			save_report = False

		if confusion == True:
			multilabel_confusion_matrix(np.array(targets), np.array(outputs_sentence) - thresholds, newEmotions, top_x=top_x, filename=f"{dataset}_unsupervised_sentence")
			multilabel_confusion_matrix(np.array(targets), np.array(outputs_centroid) - thresholds_centroids, newEmotions, top_x=top_x, filename=f"{dataset}_unsupervised_centroid")
			multilabel_confusion_matrix(np.array(targets), np.array(outputs_word) - thresholds_word, newEmotions, top_x=top_x, filename=f"{dataset}_unsupervised_word")
			confusion = False

	if len(splits) > 1:
		dev_splits = 1 - np.array(splits)
		wandb.log({
			"unsupervised_precision" : wandb.plot.line_series(
				xs=dev_splits,
				ys=[sentence_precision, word_precision, centroid_precision],
				keys=["Sentence", "Word", "Centroid"],
				title="Macro Precision",
				xname="Dev Split"),
			"unsupervised_recall" : wandb.plot.line_series(
				xs=dev_splits,
				ys=[sentence_recall, word_recall, centroid_recall],
				keys=["Sentence", "Word", "Centroid"],
				title="Macro Recall",
				xname="Dev Split"),
			"unsupervised_f1" : wandb.plot.line_series(
				xs=dev_splits,
				ys=[sentence_f1, word_f1, centroid_f1],
				keys=["Sentence", "Word", "Centroid"],
				title="Macro F1",
				xname="Dev Split"),
		})

if __name__ == '__main__':
	main()