import sklearn
from sklearn.feature_extraction.text import CountVectorizer
from analyze import *

def extractFeaturesBagOfWords(data):
	matrix = CountVectorizer(max_features=4000)
	X = matrix.fit_transform(data.text).toarray()
	return X


def main():
	data = getData()
	emotions = getEmotions()
	filtered_data = getFilteredData()
	cleanText(filtered_data)
	X = extractFeaturesBagOfWords(filtered_data)

if __name__ == "__main__":
	main()