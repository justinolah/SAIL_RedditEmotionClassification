import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from analyze import *

def extractFeaturesBagOfWords(data, ngram):
	vectorizer = CountVectorizer(max_features=4000, ngram_range=ngram)
	X = vectorizer.fit_transform(data.text).toarray()
	#print(vectorizer.vocabulary_)
	return X


def main():
	data = getFilteredData()
	cleanText(data)
	X = extractFeaturesBagOfWords(data, (1,1))

if __name__ == "__main__":
	main()