import sklearn
import liwc
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, multilabel_confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import SparsePCA, TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from helpers import *

parse, category_names = liwc.load_token_parser('data/LIWC.dic')
emoticons = getEmoticons()

#Feature extractor for LIWC Lexicon
class LIWCFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
    	return data.apply(getLIWCFeatures).to_list()

    def fit(self, X, y=None):
        return self

#Feature extractor for punctuation, emojis, and emoticons
class EmoticonsAndPunctuationExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
    	return data.apply(lambda x: getEmoticonsFeatures(x,emoticons)).to_list()

    def fit(self, X, y=None):
        return self

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, column):
        self.column = column
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.column]

class textCleaner(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def transform(self, data, y=None):
    	stopwords = getStopWords()
    	data["raw_text"] = data.text.copy()
    	data.text = data.text.apply(lambda x: processText(x,stopwords))
    	return data

    def fit(self, X, y=None):
        return self


def extractFeaturesBagOfWords(texts):
	vectorizer = CountVectorizer(min_df=3, ngram_range=(1,2))
	X = vectorizer.fit_transform(texts).toarray()
	#print(vectorizer.vocabulary_)
	#print("Vocabualry length:", len(vectorizer.vocabulary_))
	#print("Words ommited:",len(vectorizer.stop_words_))
	return X

def getEmotionIndexMap(oldEmotions, emotionMap):
	newEmotionMap = {}
	for i, (key, value) in enumerate(emotionMap.items()):
		for emotion in value:
			newEmotionMap[oldEmotions.index(emotion)] = i	
	return newEmotionMap

def getYMatrix(labels, numEmotions):
	indices = [int(label) for label in labels.split(',') if int(label) < numEmotions]
	y = list([0] * numEmotions)
	for i in indices:
		y[i] = 1
	return y

def getYMatrixWithMap(labels, numEmotions, oldidx2newidx):
	indices = [oldidx2newidx[int(label)] for label in labels.split(',') if int(label) in oldidx2newidx.keys()]
	y = list([0] * numEmotions)
	for i in indices:
		y[i] = 1
	return y

def getLIWCFeatures(text):
	words = text.split()
	counts = Counter(category for word in words for category in parse(word))
	vec = [0] * len(category_names)
	if len(words) == 0:
		return vec
	for i, name in enumerate(category_names):
		vec[i] = counts[name]/len(words)
	return vec

def getEmoticonsFeatures(text, emoticons):
	textLength = len(text.split())
	vec = [0] * len(emoticons)
	for i, emote in enumerate(emoticons):
		vec[i] = text.count(emote)/textLength
	return vec
		
def trainModel(x_train, y_train, x_test, y_test, pipeline, emotions):
	print("Training model...")
	pipeline.fit(x_train, y_train)
	prediction = pipeline.predict(x_test)
	accuracy = accuracy_score(y_test, prediction)
	#conf = multilabel_confusion_matrix(y_test, prediction)
	print("Subset Accuracy:", accuracy)
	print("")
	print(classification_report(y_test, prediction, target_names=emotions, zero_division=0))
	print("")
	#print(conf)
	print("Total Features:", len(pipeline.named_steps['clf'].coef_[0]))


def fit_hyperparameters(x_train, y_train, pipeline):
	pg = {'clf__estimator__C': [.3, 1, 3], 'feats__counts__cvec__ngram_range' : [(1,1), (1,2), (2,2)]}
	
	grid = GridSearchCV(pipeline, verbose=3, param_grid=pg, cv=5)
	grid.fit(x_train, y_train)
	print(grid.best_params_)
	print(grid.best_score_)

def svd(x_train, y_train, x_cv, y_cv):
	pass

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	test = getTestSet()
	cv = getCVSet()

	#add column of pre processed text
	cleanText(train)
	cleanText(test)
	cleanText(cv)

	print("Training set length:", len(train))
	print("Testing set length:", len(test))
	print("Cross validation set length:", len(cv))
	print("")

	liwcPipe = Pipeline([('selector', ColumnSelector(column='text')), ('liwc', LIWCFeatureExtractor())])
	cvecPipe = Pipeline([('selector', ColumnSelector(column='text')), ('cvec', CountVectorizer(min_df=3, binary=True, ngram_range=(1,2)))])
	tfidPipe = Pipeline([('selector', ColumnSelector(column='text')), ('tfid', TfidfVectorizer(min_df=3, ngram_range=(1,2)))])
	emotesPipe = Pipeline([('selector', ColumnSelector(column='raw_text')), ('emot', EmoticonsAndPunctuationExtractor())])

	features = FeatureUnion([
		('lex', liwcPipe),
		('counts', cvecPipe),
		('emotes', emotesPipe),
    ])

	features_tfid = FeatureUnion([
    	('lex', liwcPipe),
		('counts', tfidPipe),
		('emotes', emotesPipe),
    ])

	logRegPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'), n_jobs=1)),
	])

	logRegPipeline_tsvd = Pipeline([
		('feats', features),
		('tsvd', TruncatedSVD(n_components=8509)),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'), n_jobs=1)),
	])

	logRegPipeline_tfid = Pipeline([
		('feats', features_tfid),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'), n_jobs=1)),
	])

	linSVCPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LinearSVC(max_iter=3000))),
	])

	linSVCPipeline_tfid = Pipeline([
		('feats', features_tfid),
		('clf', OneVsRestClassifier(LinearSVC(max_iter=3000))),
	])

	x_train = train
	x_test = test

	y_train = train.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()
	y_test = test.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()
	y_cv = cv.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()

	ekmanDict = getEkmanDict()
	sentDict = getSentimentDict()
	ekEmotions = ekmanDict.keys()
	sentEmotions = sentDict.keys()
	ek_idx_map = getEmotionIndexMap(emotions, ekmanDict)
	sent_idx_map = getEmotionIndexMap(emotions, sentDict)

	y_train_ek = train.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions),ek_idx_map)).to_list()
	y_test_ek = test.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions), ek_idx_map)).to_list()
	y_cv_ek = cv.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions), ek_idx_map)).to_list()

	y_train_sent = train.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions),sent_idx_map)).to_list()
	y_test_sent = test.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions), sent_idx_map)).to_list()
	y_cv_sent = cv.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions), sent_idx_map)).to_list()

	pipeline = logRegPipeline

	#fit_hyperparameters(x_train, y_train, pipeline)
	trainModel(x_train, y_train, x_test, y_test, pipeline, emotions)
	trainModel(x_train, y_train_sent, x_test, y_test_sent, pipeline, sentEmotions)
	trainModel(x_train, y_train_ek, x_test, y_test_ek, pipeline, ekEmotions)
	return

	#cross validation
	all_models = [
    	("log", logRegPipeline),
    	("log_tfid", logRegPipeline_tfid),
    	("lsvc", linSVCPipeline),
    	("lsvc_tfid", linSVCPipeline_tfid),
    ]
 
	scores = [(name, cross_val_score(model, x_train, y_train, cv=5, verbose=3).mean()) for name, model in all_models]
	print(scores)

	tsvd_options = [8509, 4254, 2127, 1063, 531]




if __name__ == "__main__":
	main()