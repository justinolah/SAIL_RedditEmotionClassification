import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from empath import Empath
import string
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
STOP_WORDS_FILE = "data/stopwords.txt"
LEXICON = "data/NRC-Emotion-Lexicon.csv"
MIN_WORD_LENGTH = 2

lemmatizer = WordNetLemmatizer()

punct_chars = list((set(string.punctuation) | {
    "’", "‘", "–", "—", "~", "|", "“", "”", "…", "'", "`", "_",
    "“", "´", "♀️", "♂"
}))
punct_chars.sort()
punctuation = "".join(punct_chars)
replace = re.compile("[%s]" % re.escape(punctuation))


def processText(text, stopwords):
	text = text.lower()
	text = re.sub(r"[0-9]", " ", text)
	text = replace.sub(" ", text)
	text = re.sub(r"\s+", " ", text)
	text = text.strip()
	words = text.split()
	return " ".join([lemmatizer.lemmatize(word) for word in words if len(word) > MIN_WORD_LENGTH and word not in stopwords])

def cleanText(data):
	stopwords = getStopWords()
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
		emotions = f.read().splitlines()
	return emotions

def getTrainSet():
	return pd.read_csv(TRAIN_SET_FILE)

def getTestSet():
	return pd.read_csv(TEST_SET_FILE)

def main():
	data = getData()

	print("Total examples:", len(data))

	#remove unclear examples
	data = data[data["example_very_unclear"] == False]

	print("Total examples with annotations:", len(data))
	print("Total unique text entries:", data.id.nunique())

	emotions = getEmotions()
	print("Total Emotions:", len(emotions))
	print("")

	print("Number of annotations:")
	for emotion in emotions:
		print(emotion, len(data[data[emotion] == 1]))
	print("")

	#filter data to only keep annotations that at least 2 raters agreed on
	filtered_data = getFilteredData()
	print("Total number of examples post agreement filter:", len(filtered_data))
	print("Number of annotations post agreement filter:")
	for emotion in emotions:
		print(emotion, len(filtered_data[filtered_data[emotion] == 1]))
	print("")

	#Text length analysis
	text_lengths = data.text.drop_duplicates().str.split().str.len()

	print("Text Length Statistics:")
	print(text_lengths.describe())
	print("")

	text_lengths = text_lengths.tolist()
	plt.hist(text_lengths, bins=32)
	plt.xlim(0,35)
	plt.ylabel('Total')
	plt.xlabel('Text Lengths')
	plt.title("Distribution of Text Lengths")
	plt.savefig("plots/length_distributions.pdf", format="pdf")

	print("Average text length:")
	for emotion in emotions:
		texts = data[data[emotion] == 1].text
		print(emotion, round(texts.str.split().str.len().sum()/len(texts),2))
	print("")

	#get list of texts that have less than 3 words
	short_texts = [text for text in set(data.text) if len(text.split()) < 3]
	with open("tables/short_texts.txt","w") as f:
		f.write('\n'.join(short_texts))


	#correlation coefficents
	correlation = data.groupby("id")[emotions].mean().corr() #calculate mean for duplicate text entries

	fig, _ = plt.subplots(figsize=(11, 9))

	palette = sns.diverging_palette(220, 20, n=256)

	sns.heatmap(
    	correlation, 
    	vmin=-.25,
    	vmax=.25, 
    	center=0,
    	cmap=palette,
    	square=True
	)

	fig.savefig("plots/correlation_coef.pdf", format="pdf")

	#analysis of highly correlated emotions
	print("Overlap of emotion labels for emotion pairs with corelation greater than .1")
	for emotion1 in emotions:
		for emotion2 in emotions:
			cor = correlation.at[emotion1, emotion2]
			if emotion1 != emotion2 and cor > .1:
				num_emotion1 = len(data[data[emotion1] == 1])
				num_overlap = len(data[(data[emotion1] == 1) & (data[emotion2] == 1)])
				perc = round(num_overlap/num_emotion1, 2)
				print(f'{perc} of {emotion1} also has {emotion2}. (corr: {round(cor, 2)})')
	print("")

	#Process text to remove punctuation, extra whitespace, etc.
	cleanText(filtered_data)

	#Get frequency of words
	word_freq = pd.Series(' '.join(filtered_data.text).split()).value_counts()
	word_freq.to_csv("tables/word_frequency.csv")
	print("Most frequent words:")
	print(word_freq[:10])
	print("")

	#Get vocab list of words in text examples
	vocab_list = sorted(word_freq[word_freq > 2].index)
	print("Total unique words that occure more than once:", len(vocab_list))

	with open("tables/vocab_list.txt","w") as f:
		f.write('\n'.join(vocab_list))

	empath = Empath()
	print("Empath lexicon total categories:", len(empath.analyze("Hell World")))

	lexicon = getLexicon()
	vocab_list = [word for word in vocab_list if word in lexicon]
	print("Total words that are in NRC emotion lexicon:", len(vocab_list))





if __name__ == "__main__":
	main()