import sklearn
import liwc
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, multilabel_confusion_matrix, precision_recall_curve
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
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
	confusionMatrices = multilabel_confusion_matrix(y_test, prediction)
	print("Total Features:", len(pipeline.named_steps['clf'].coef_[0]))
	print("Subset Accuracy:", accuracy)
	print("")
	print(classification_report(y_test, prediction, target_names=emotions, zero_division=0))
	print("")
	for i, emotion in enumerate(emotions):
		print(f"{emotion}:")
		print(pd.DataFrame(confusionMatrices[i], columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))

def fit_hyperparameters(x_train, y_train, x_cv, y_cv, pipeline):
	pg = {'clf__estimator__C': [.01, .1, .3, 1, 3, 10, 30, 100]}
	scorers = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']

	split = [(list(range(len(x_train))), list(range(len(x_train), len(x_train) + len(x_cv))))] 
	
	grid = GridSearchCV(pipeline, verbose=3, param_grid=pg, refit='f1_micro', cv=split, scoring=scorers)
	grid.fit(x_train.append(x_cv), y_train + y_cv)
	print(grid.best_params_)
	print(grid.best_score_)

def crossValidateModels(x_train, y_train, x_cv, y_cv, models):
	split = [(list(range(len(x_train))), list(range(len(x_train), len(x_train) + len(x_cv))))] 
	scores = [(name, cross_val_score(model, x_train.append(x_cv), y_train + y_cv, scoring='f1_micro', cv=split, verbose=3).mean()) for name, model in models]
	print(scores)


def analyzeThresholds(pipeline, x_test, y_test, emotions):
	y_scores = pipeline.predict_proba(x_test)
	y_test = np.array(y_test)

	for i, emotion in enumerate(emotions):
		p, r, thresholds = precision_recall_curve(y_test[:,i], y_scores[:,i])
		plt.figure(figsize=(8, 8))
		plt.title(f"{emotion}: Precision vs Recall")
		plt.plot(r, p)
		plt.ylabel("Recall")
		plt.xlabel("Precision")
		plt.show()

		plt.figure(figsize=(8, 8))
		plt.title(f"{emotion}: threshold vs precision, recall, and f1")
		plt.plot(thresholds, p[:-1], "b--", label="Precision")
		plt.plot(thresholds, r[:-1], "g-", label="Recall")
		plt.plot(thresholds, 2 * p[:-1] * r[:-1] / (r[:-1] + p[:-1]), "r-", label="F1")
		plt.ylabel("Score")
		plt.xlabel("Decision Threshold")
		plt.legend(loc='best')
		plt.show()

def svd(x_train, y_train, x_cv, y_cv, features, emotions):
	tsvd_options = [8509, 4254, 2127, 1063, 531]

	for num in tsvd_options:
		logRegPipeline_tsvd = Pipeline([
			('feats', features),
			('tsvd', TruncatedSVD(n_components=num)),
			('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'))),
		])
		print("Features:", num)
		trainModel(x_train, y_train, x_cv, y_cv, logRegPipeline_tsvd, emotions)


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
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', C=1.5))),
	])

	logRegPipeline_tfid = Pipeline([
		('feats', features_tfid),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs', C=10))),
	])

	linSVCPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=3000, C=0.1))),
	])

	linSVCPipeline_tfid = Pipeline([
		('feats', features_tfid),
		('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=3000, C=1))),
	])

	MNBPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(MultinomialNB(alpha=1.0))),
	])

	x_train = train
	x_test = test
	x_cv = cv

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

	svd(x_train, y_train, x_cv, y_cv, features, emotions)
	return

	#fit_hyperparameters(x_train, y_train, x_cv, y_cv, pipeline)

	trainModel(x_train, y_train, x_test, y_test, pipeline, emotions)
	print("Sentiment Grouping:")
	trainModel(x_train, y_train_sent, x_test, y_test_sent, pipeline, sentEmotions)
	print("Ekman Grouping:")
	trainModel(x_train, y_train_ek, x_test, y_test_ek, pipeline, ekEmotions)

	#analyzeThresholds(pipeline, x_test, y_test, emotions)

	return
	#cross validation
	models = [
    	("log", logRegPipeline),
    	("log_tfid", logRegPipeline_tfid),
    	("lsvc", linSVCPipeline),
    	("lsvc_tfid", linSVCPipeline_tfid),
    ]
 
	crossValidateModels(x_train, y_train, x_cv, y_cv, models)



if __name__ == "__main__":
	main()