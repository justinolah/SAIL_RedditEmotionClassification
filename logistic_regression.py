import sklearn
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV
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
    	return pd.Series([processText(text,stopwords) for text in data])

    def fit(self, X, y=None):
        return self


def extractFeaturesBagOfWords(texts):
	vectorizer = CountVectorizer(min_df=3, ngram_range=(1,2))
	X = vectorizer.fit_transform(texts).toarray()
	print(vectorizer.vocabulary_)
	print("Vocabualry length:", len(vectorizer.vocabulary_))
	print("Words ommited:",len(vectorizer.stop_words_))
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
		
def trainModel(x_train, y_train, x_test, y_test, pipeline, emotions):
	print("Training model...")
	pipeline.fit(x_train, y_train)
	prediction = pipeline.predict(x_test)
	accuracy = accuracy_score(y_test, prediction)
	print("Subset Accuracy:", accuracy)
	print("")
	print(classification_report(y_test, prediction, target_names=emotions,zero_division=0))

	#print("Neutral:", len([row for row in prediction if np.sum(row) == 0]))

def fit_hyperparameters(x_train, y_train, pipeline):
	pg = {'clf__estimator__C': [.3, 1, 3], 'feats__cvec__ngram_range' : [(1,1), (1,2), (2,2)]}
	
	grid = GridSearchCV(pipeline, verbose=3, param_grid=pg, cv=5)
	grid.fit(x_train, y_train)
	print(grid.best_params_)
	print(grid.best_score_)

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	train = getTrainSet()
	test = getTestSet()

	features = FeatureUnion([
		('liwc', LIWCFeatureExtractor()),
    	('cvec', CountVectorizer(min_df=3, binary=True, ngram_range=(1,2))),
    ])

	features_tfid = FeatureUnion([
    	('liwc', LIWCFeatureExtractor()),
    	('tfid', TfidfVectorizer(min_df=3, ngram_range=(1,1))),
    ])

	logRegPipeline = Pipeline([
		('selector', ColumnSelector('text')),
		('cleaner', textCleaner()),
		('feats', features),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'), n_jobs=1)),
	])

	logRegPipeline_tfid = Pipeline([
		('selector', ColumnSelector('text')),
		('cleaner', textCleaner()),
		('feats', features_tfid),
		('clf', OneVsRestClassifier(LogisticRegression(penalty='l2', max_iter=1000, solver='lbfgs'), n_jobs=1)),
	])

	x_train = train
	x_test = test

	y_train = train.labels.apply(lambda x: getYVector(x,len(emotions))).to_list()
	y_test = test.labels.apply(lambda x: getYVector(x,len(emotions))).to_list()

	fit_hyperparameters(x_train, y_train, logRegPipeline)
	trainModel(x_train, y_train, x_test, y_test, logRegPipeline, emotions)




if __name__ == "__main__":
	main()