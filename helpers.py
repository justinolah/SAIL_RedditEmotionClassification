import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re 
import os
import json
import itertools
from os import path

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer

CSV_FILES = ["data/full_dataset/goemotions_1.csv", "data/full_dataset/goemotions_2.csv", "data/full_dataset/goemotions_3.csv"]
FILTERED_DATA_FILE = "data/filtered_data.csv" 
EMOTION_FILE = "data/emotions.txt"
TRAIN_SET_FILE = "data/train.csv"
TEST_SET_FILE = "data/test.csv"
VAL_SET_FILE = "data/dev.csv"
TRAIN_EMPATH_FILE = "data/train_empath_features.csv"
TEST_EMPATH_FILE = "data/test_empath_features.csv"
VAL_EMPATH_FILE = "data/dev_empath_features.csv"
STOP_WORDS_FILE = "data/stopwords.txt"
EMOTICONS_FILE = "data/emoticons.txt"
LEXICON = "data/NRC-Emotion-Lexicon.csv"
SENTIMENT_DICT_FILE = "data/sentiment_map.json"
EKMAN_DICT_FILE = "data/ekman_map.json"
MIN_WORD_LENGTH = 3

lemmatizer = WordNetLemmatizer()

def processText(text, stopwords):
	text = text.lower()
	text = re.sub(r"[^a-z\s]+", " ", text)
	text = re.sub(r"\s+", " ", text)
	text = text.strip()
	words = text.split()
	return " ".join([lemmatizer.lemmatize(word) for word in words if len(word) >= MIN_WORD_LENGTH and word not in stopwords])

def cleanText(data):
	stopwords = getStopWords()
	data["raw_text"] = data.text.copy()
	data.text = data.text.apply(lambda x: processText(x,stopwords))
	data = data[data.text.str.len() > 0]

def CheckAgreement(ex, min_agreement, all_emotions, max_agreement=100):
	sum_ratings = ex[all_emotions].sum(axis=0)
	agreement = ((sum_ratings >= min_agreement) & (sum_ratings <= max_agreement))
	return ",".join(sum_ratings.index[agreement].tolist())

def getData():
	data = []
	for csv in CSV_FILES:
		data.append(pd.read_csv(csv))
	return pd.concat(data)

def getFilteredData():
	return pd.read_csv(FILTERED_DATA_FILE)

def getStopWords():
	with open(STOP_WORDS_FILE) as f:
		stopwords = f.read().splitlines()
	return stopwords

def getLexicon():
	return pd.read_csv(LEXICON).words.to_list()

def getEmotions():
	with open(EMOTION_FILE) as f:
		return f.read().splitlines()

#list of emoticons, $, ?, and !
def getEmoticons():
	with open(EMOTICONS_FILE) as f:
		return f.read().splitlines()

def getTrainSet():
	return pd.read_csv(TRAIN_SET_FILE)

def getTestSet():
	return pd.read_csv(TEST_SET_FILE)

def getValSet():
	return pd.read_csv(VAL_SET_FILE)

def getTrainEmpath(data):
	empath = pd.read_csv(TRAIN_EMPATH_FILE).empath
	data["empath"] = empath
	return data

def getTestEmpath(data):
	empath = pd.read_csv(TEST_EMPATH_FILE).empath
	data["empath"] = empath
	return data

def getValEmpath(data):
	empath = pd.read_csv(VAL_EMPATH_FILE).empath
	data["empath"] = empath
	return data

def getSentimentDict():
	with open(SENTIMENT_DICT_FILE) as json_file:
		return json.load(json_file)

def getEkmanDict():
	with open(EKMAN_DICT_FILE) as json_file:
		return json.load(json_file)

def getEmotionIndexMap(oldEmotions, emotionMap):
	newEmotionMap = {}
	for i, (key, value) in enumerate(emotionMap.items()):
		for emotion in value:
			newEmotionMap[oldEmotions.index(emotion)] = i	
	return newEmotionMap

def get_topx_inference(pred_arr, top_x=1):
	pred_at_topx = pred_arr.copy()
	for n in range(pred_arr.shape[0]):
		pred_n = pred_arr[n]
		pred_at_topx[n][np.argsort(pred_n)[-top_x:].tolist()] = 1
		pred_at_topx[pred_at_topx!=1] = 0
	return pred_at_topx   

def multilabel_confusion_matrix(gt, pred, emotions, top_x=1, filename="model"):
	num_samples, num_labels = gt.shape
	cm = np.zeros((num_labels, num_labels))
	pred_at_top_x = get_topx_inference(pred, top_x)
	for i in range(num_samples):
		where_gt = np.where(gt[i])[0].tolist()
		where_pred = np.where(pred_at_top_x[i])[0].tolist()
		gt_v_pred = list(itertools.product(where_gt, where_pred))
		for m,n in gt_v_pred: cm[m,n]+=1
	cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

	fig, _ = plt.subplots(figsize=(11, 9))
	palette = sns.diverging_palette(220, 20, n=256)
	sns.heatmap(
		cm, 
		vmin=0,
		vmax=.5, 
		center=0,
		xticklabels=emotions, 
		yticklabels=emotions,
		cmap=palette,
		square=True
	)
	plt.xlabel('Predicted')
	plt.ylabel('Actual')
	fig.savefig("plots/" + filename + "_confusion_matrix.pdf", format="pdf")

	return cm 

def svdResultsGraph():
	num_features = [8509, 4254, 2127, 1063, 531, 256, 132, 66]
	accuracy = [.4710652414301511, .4677478805750092, .46737928492443787, .4659049023221526, .4552156284555842, .4345742720235901, .41467010689273864, .3964246221894582]
	micro_prec = [.72, .71, .72, .73, .73, .73, .74, .77]
	micro_rec = [.34, .34, .33, .32, .30, .26, .22, .19]
	micro_f1 = [.46, .46, .45, .45, .43, .39, .34, .30]
	macro_prec = [.66, .65, .64, .64, .56, .46, .36, .28]
	macro_rec = [.28, .28, .27, .26, .23, .18, .15, .12]
	macro_f1 = [.36, .36, .35, .34, .30, .23, .19, .15]

	plt.figure(figsize=(8, 8))
	plt.title("SVD Results")
	plt.plot(range(1,9), accuracy, "k--", label="Accuracy")
	plt.plot(range(1,9), micro_prec, "g--", label="Micro prec")
	plt.plot(range(1,9), micro_rec, "b--", label="Micro rec")
	plt.plot(range(1,9), micro_f1, "c--", label="Micro f1")
	plt.plot(range(1,9), macro_prec, "m--", label="Macro prec")
	plt.plot(range(1,9), macro_rec, "y--", label="Macro rec")
	plt.plot(range(1,9), macro_f1, "r--", label="Macro f1")
	plt.ylabel("Score")
	plt.xlabel(".5^n features")
	plt.legend(loc='best')
	plt.savefig("plots/svd_results.pdf", format="pdf")
