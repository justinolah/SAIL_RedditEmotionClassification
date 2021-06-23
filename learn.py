import sklearn
import liwc
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
import xgboost as xgb
from xgboost import XGBClassifier
from helpers import *

parse, category_names = liwc.load_token_parser('data/LIWC.dic')
emoticons = getEmoticons()

xgb.set_config(verbosity=2)

#Feature extractor for LIWC Lexicon
class LIWCFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, binary=False):
        self.binary = binary

    def transform(self, data, y=None):
    	return data.apply(lambda x: getLIWCFeatures(x,self.binary)).to_list()

    def fit(self, X, y=None):
        return self

#Feature extractor for punctuation, emojis, and emoticons
class EmoticonsAndPunctuationExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, binary=False):
        self.binary = binary

    def transform(self, data, y=None):
    	return data.apply(lambda x: getEmoticonsFeatures(x,emoticons, self.binary)).to_list()

    def fit(self, X, y=None):
        return self

class EmpathExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, binary=False):
        self.binary = binary

    def transform(self, data, y=None):
    	feats = []
    	for row in data:
    		row = row.replace("[","")
    		row = row.replace("]","")
    		row = [int(bool(float(item))) if True else float(item) for item in row.split(',')]
    		feats.append(row)
    	return feats

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

def getLIWCFeatures(text, binary=False):
	words = text.split()
	counts = Counter(category for word in words for category in parse(word))
	vec = [0] * len(category_names)
	if len(words) == 0:
		return vec
	for i, name in enumerate(category_names):
		vec[i] = int(bool(counts[name])) if binary else counts[name]/len(words)
	return vec

def getEmoticonsFeatures(text, emoticons, binary=False):
	textLength = len(text.split())
	vec = [0] * len(emoticons)
	for i, emote in enumerate(emoticons):
		vec[i] = int(emote in text) if binary else text.count(emote)/textLength
	return vec

def validateModels(x_train, y_train, x_val, y_val, models):
	split = [(list(range(len(x_train))), list(range(len(x_train), len(x_train) + len(x_val))))] 
	scores = [(name, cross_val_score(model, x_train.append(x_val), y_train + y_val, scoring='f1_macro', cv=split, verbose=3).mean()) for name, model in models]
	print(scores)


def analyzeThresholds(pipeline, x_test, y_test, emotions):
	y_scores = pipeline.predict_proba(x_test)
	y_test = np.array(y_test)

	for i, emotion in enumerate(emotions):
		p, r, thresholds = precision_recall_curve(y_test[:,i], y_scores[:,i])
		plt.figure(figsize=(8, 8))
		plt.title(f"{emotion}: threshold vs precision, recall, and f1")
		plt.plot(thresholds, p[:-1], "b--", label="Precision")
		plt.plot(thresholds, r[:-1], "g-", label="Recall")
		plt.plot(thresholds, 2 * p[:-1] * r[:-1] / (r[:-1] + p[:-1]), "r-", label="F1")
		plt.ylabel("Score")
		plt.xlabel("Decision Threshold")
		plt.legend(loc='best')
		plt.show()

def svd(x_train, y_train, x_val, y_val, features, emotions):
	tsvd_options = [8509, 4254, 2127, 1063, 531]

	for num in tsvd_options:
		logRegPipeline_tsvd = Pipeline([
			('feats', features),
			('tsvd', TruncatedSVD(n_components=num)),
			('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'))),
		])
		print("Features:", num)
		trainModel(x_train, y_train, x_val, y_val, logRegPipeline_tsvd, emotions)

def trainModel(x_train, y_train, x_test, y_test, pipeline, emotions, filename="model"):
	print("Training model...")
	pipeline.fit(x_train, y_train)

	prediction = pipeline.predict(x_test)
	prediction_probabilities = pipeline.predict_proba(x_test)
	accuracy = accuracy_score(y_test, prediction)
	#print("Total Features:", len(pipeline.named_steps['clf'].coef_[0]))
	print("Subset Accuracy:", accuracy)
	print(classification_report(y_test, prediction, target_names=emotions, zero_division=0, output_dict=False))
	report = classification_report(y_test, prediction, target_names=emotions, zero_division=0, output_dict=True)

	#export resuls to csv
	micro = list(report['micro avg'].values())
	micro.pop()
	macro = list(report['macro avg'].values())
	macro.pop()
	scores = [accuracy, *micro, *macro]
	results = pd.DataFrame(data=[scores], columns=['accuracy', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'])
	results.to_csv("tables/" + filename + "_results.csv")

	#confusion matrix
	multilabel_confusion_matrix(np.array(y_test), np.array(prediction_probabilities), emotions, top_x=3, filename=filename)

def randomFitHyperparameters(x_train, y_train, x_val, y_val, pipeline):
	pg = {
			#'clf__estimator__max_depth': [500],
			#'clf__estimator__min_samples_split': [2, 5, 10],
			#'clf__estimator__min_samples_leaf': [2, 5, 10],
			'clf__estimator__n_estimators':  [100],
			#'clf__estimator__max_samples':  [.1, .2, .4],
			'clf__estimator__max_features':  ['sqrt'],
		}
	scorers = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']

	split = [(list(range(len(x_train))), list(range(len(x_train), len(x_train) + len(x_val))))] 
	
	grid = RandomizedSearchCV(pipeline, param_distributions=pg, verbose=3, n_iter=10, refit='f1_macro', random_state=17, cv=split, scoring=scorers)
	grid.fit(x_train.append(x_val), y_train + y_val)
	print(grid.best_params_)
	print(grid.best_score_)

	#export results to csv
	params = grid.cv_results_['params']
	results = [params]
	for scorer in scorers:
		results.append(grid.cv_results_['mean_test_' + scorer])

	scorers.append('mean_fit_time')
	results.append(grid.cv_results_['mean_fit_time'])

	results = list(zip(*results))
	scorers.insert(0,"params")
	results = pd.DataFrame(data=results, columns=scorers)
	results.to_csv("tables/random_hyperparameter_results.csv")

def fitHyperparameters(x_train, y_train, x_val, y_val, pipeline):
	pg = [
		{
			'clf__estimator__max_depth': [500],
			'clf__estimator__max_features':  ['sqrt'],
			'clf__estimator__n_estimators':  [100],
			'clf__estimator__max_samples':  [.75,],
			'clf__estimator__min_samples_split': [1, 2, 5],
			'clf__estimator__min_samples_leaf': [1, 2, 5],
		},
	]
	scorers = ['accuracy', 'precision_micro', 'recall_micro', 'f1_micro', 'precision_macro', 'recall_macro', 'f1_macro']

	split = [(list(range(len(x_train))), list(range(len(x_train), len(x_train) + len(x_val))))] 
	
	grid = GridSearchCV(pipeline, verbose=3, param_grid=pg, refit='f1_macro', cv=split, scoring=scorers)
	grid.fit(x_train.append(x_val), y_train + y_val)
	print(grid.best_params_)
	print(grid.best_score_)

	#export results to csv
	params = grid.cv_results_['params']
	results = [params]
	for scorer in scorers:
		results.append(grid.cv_results_['mean_test_' + scorer])

	scorers.append('mean_fit_time')
	results.append(grid.cv_results_['mean_fit_time'])

	results = list(zip(*results))
	scorers.insert(0,"params")
	results = pd.DataFrame(data=results, columns=scorers)
	results.to_csv("tables/hyperparameter_results.csv")


def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	test = getTestSet()
	val = getValSet()

	#add column of pre processed text
	cleanText(train)
	cleanText(test)
	cleanText(val)

	#get saved empath features
	getTrainEmpath(train)
	getTestEmpath(test)
	getValEmpath(val)

	print("Training set length:", len(train))
	print("Testing set length:", len(test))
	print("Cross validation set length:", len(val))
	print("")

	liwcPipe = Pipeline([('selector', ColumnSelector(column='text')), ('liwc', LIWCFeatureExtractor())])
	liwcPipeBinary = Pipeline([('selector', ColumnSelector(column='text')), ('liwc', LIWCFeatureExtractor(binary=True))])
	cvecPipe = Pipeline([('selector', ColumnSelector(column='text')), ('cvec', CountVectorizer(min_df=3, binary=True, ngram_range=(1,2)))])
	tfidfPipe = Pipeline([('selector', ColumnSelector(column='text')), ('tfidf', TfidfVectorizer(min_df=3, ngram_range=(1,2)))])
	emotesPipe = Pipeline([('selector', ColumnSelector(column='raw_text')), ('emot', EmoticonsAndPunctuationExtractor())])
	emotesPipeBinary = Pipeline([('selector', ColumnSelector(column='raw_text')), ('emot', EmoticonsAndPunctuationExtractor(binary=True))])
	empathPipe = Pipeline([('selector', ColumnSelector(column='empath')), ('emp', EmpathExtractor(binary=False))])
	empathPipeBinary = Pipeline([('selector', ColumnSelector(column='empath')), ('emp', EmpathExtractor(binary=True))])

	features = FeatureUnion([
		('lex', liwcPipe),
		('counts', cvecPipe),
		('emotes', emotesPipeBinary),
    ])

	features_tfidf = FeatureUnion([
    	('lex', liwcPipe),
		('counts', tfidfPipe),
		('emotes', emotesPipe),
    ])

	logRegPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, C=1, class_weight='balanced'))),
	])

	logRegPipeline_tfidf = Pipeline([
		('feats', features_tfidf),
		('clf', OneVsRestClassifier(LogisticRegression(max_iter=1000, C=5, class_weight='balanced'))),
	])

	linSVCPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=2500, C=1, penalty='l1', class_weight=None))),
	])

	linSVCPipeline_tfidf = Pipeline([
		('feats', features_tfidf),
		('clf', OneVsRestClassifier(LinearSVC(dual=False, max_iter=2500, C=0.1, penalty='l2', class_weight='balanced'))),
	])

	ridgePipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(RidgeClassifier(alpha=0.25, class_weight=None))),
	])

	ridgePipeline_tfidf = Pipeline([
		('feats', features_tfidf),
		('clf', OneVsRestClassifier(RidgeClassifier(alpha=0.1, class_weight=None))),
	])

	#does not work
	lassoPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(Lasso())),
	])

	rforestPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(RandomForestClassifier(random_state=42, class_weight=None))),
	])

	xgboostPipeline = Pipeline([
		('feats', features),
		('clf', OneVsRestClassifier(XGBClassifier())),
	])

	x_train = train
	x_test = test
	x_val = val

	y_train = train.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()
	y_test = test.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()
	y_val = val.labels.apply(lambda x: getYMatrix(x,len(emotions))).to_list()

	ekmanDict = getEkmanDict()
	sentDict = getSentimentDict()
	ekEmotions = ekmanDict.keys()
	sentEmotions = sentDict.keys()
	ek_idx_map = getEmotionIndexMap(emotions, ekmanDict)
	sent_idx_map = getEmotionIndexMap(emotions, sentDict)

	y_train_ek = train.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions),ek_idx_map)).to_list()
	y_test_ek = test.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions), ek_idx_map)).to_list()
	y_val_ek = val.labels.apply(lambda x: getYMatrixWithMap(x,len(ekEmotions), ek_idx_map)).to_list()

	y_train_sent = train.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions),sent_idx_map)).to_list()
	y_test_sent = test.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions), sent_idx_map)).to_list()
	y_val_sent = val.labels.apply(lambda x: getYMatrixWithMap(x,len(sentEmotions), sent_idx_map)).to_list()

	pipeline = rforestPipeline

	#svd(x_train, y_train, x_val, y_val, features, emotions)
	fitHyperparameters(x_train, y_train, x_val, y_val, pipeline)
	#randomFitHyperparameters(x_train, y_train, x_val, y_val, pipeline)
	return
	trainModel(x_train, y_train, x_test, y_test, pipeline, emotions)
	return

	#analyzeThresholds(pipeline, x_cv, y_cv, emotions)
	print("Sentiment Grouping:")
	trainModel(x_train, y_train_sent, x_test, y_test_sent, pipeline, sentEmotions, "sentiment")
	print("Ekman Grouping:")
	trainModel(x_train, y_train_ek, x_test, y_test_ek, pipeline, ekEmotions, "ekman")

	return

	#cross validation
	models = [
    	("log", logRegPipeline),
    	("log_tfidf", logRegPipeline_tfidf),
    	("lsvc", linSVCPipeline),
    	("lsvc_tfidf", linSVCPipeline_tfidf),
    ]
 
	validateModels(x_train, y_train, x_val, y_val, models)




if __name__ == "__main__":
	main()