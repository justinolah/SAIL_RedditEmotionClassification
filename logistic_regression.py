import sklearn
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from analyze import *

import liwc
parse, category_names = liwc.load_token_parser('data/LIWC.dic')

#Feature extractor for LIWC Lexicon
class LIWCFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
    	return data.apply(getLIWCFeatures).to_list()

    def fit(self, X, y=None):
        return self

def extractFeaturesBagOfWords(texts, vocab_list):
	vectorizer = CountVectorizer(max_features=4000, vocabulary=vocab_list, ngram_range=(1,1))
	X = vectorizer.fit_transform(texts).toarray()
	#print(vectorizer.vocabulary_)
	#print(len(vectorizer.vocabulary_))
	return X

def getYVector(labels, numEmotions):
	indecies = [int(label) for label in labels.split(',') if int(label) < numEmotions] #removes neutral tag
	y = [0] * numEmotions
	for i in indecies:
		y[i] = 1
	return list(y)

def getLIWCFeatures(text):
	words = text.split()
	counts = Counter(category for word in words for category in parse(word))
	vec = [0] * len(category_names)
	if len(words) == 0:
		return vec
	for i, name in enumerate(category_names):
		vec[i] = counts[name]/len(words)
	return vec
		
def trainLogisticRegression(X,y):
	model = LogisticRegression(verbose=1, solver='liblinear',random_state=0, C=5, penalty='l2',max_iter=100).fit(X,y)

def main():
	data = getFilteredData()
	cleanText(data)

	word_freq = pd.Series(' '.join(data.text).split()).value_counts()
	vocab_list = sorted(word_freq[word_freq > 2].index)
	print(len(vocab_list))

	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	cleanText(train)
	
	test = getTestSet()
	cleanText(test)

	features = FeatureUnion([
		('liwc', LIWCFeatureExtractor()),
    	('cvec', CountVectorizer(vocabulary=vocab_list))
    ])

	logRegPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, solver='sag'), n_jobs=1)),
	])

	x_train = train.text
	x_test = test.text

	y_train = train.labels.apply(lambda x: getYVector(x,len(emotions))).to_list()
	y_test = test.labels.apply(lambda x: getYVector(x,len(emotions))).to_list()

	print("Training model...")
	logRegPipeline.fit(x_train, y_train)
	prediction = logRegPipeline.predict(x_test)
	accuracy = accuracy_score(y_test, prediction)
	precision, recall, fscore, _ = precision_recall_fscore_support(y_test, prediction, zero_division=0)

	print("Accuracy:", accuracy)
	for i, emotion in enumerate(emotions):
		print(emotion)
		print("precision:", precision[i])
		print("Recall:", recall[i])
		print("F1 Score:", fscore[i])
		print("")



if __name__ == "__main__":
	main()