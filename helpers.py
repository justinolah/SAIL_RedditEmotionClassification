import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import re 
import os
from os import path

import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer, PorterStemmer

CSV_FILES = ["data/full_dataset/goemotions_1.csv", "data/full_dataset/goemotions_2.csv", "data/full_dataset/goemotions_3.csv"]
FILTERED_DATA_FILE = "data/filtered_data.csv" 
EMOTION_FILE = "data/emotions.txt"
TRAIN_SET_FILE = "data/train.csv"
TEST_SET_FILE = "data/test.csv"
CV_SET_FILE = "data/dev.csv"
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

def getCVSet():
	return pd.read_csv(CV_SET_FILE)

def getSentimentDict():
	with open(SENTIMENT_DICT_FILE) as json_file:
		return json.load(json_file)

def getEkmanDict():
	with open(EKMAN_DICT_FILE) as json_file:
		return json.load(json_file)
