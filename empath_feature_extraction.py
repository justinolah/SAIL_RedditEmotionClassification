from empath import Empath
from analyze import *

empath = Empath()
empathLength = len(empath.analyze("Hello"))

def getEmpathFeatures(text):
	categories = empath.analyze(text, normalize=True)
	if categories is None:
		return [0] * empathLength
	else:
		return list(categories.values())

def main():
	train = getTrainSet()
	cleanText(train)

	test = getTestSet()
	cleanText(test)

	val = getValSet()
	cleanText(val)

	print("Extracting empath features from train set...")
	empathFeatures_train = train.text.apply(getEmpathFeatures).rename("empath")
	empathFeatures_train.to_csv("data/train_empath_features.csv")
	print("")
	print("Extracting empath features from test set...")
	empathFeatures_test = test.text.apply(getEmpathFeatures).rename("empath")
	empathFeatures_test.to_csv("data/test_empath_features.csv")
	print("")
	print("Extracting empath features from validation set...")
	empathFeatures_val = val.text.apply(getEmpathFeatures).rename("empath")
	empathFeatures_val.to_csv("data/dev_empath_features.csv")


if __name__ == "__main__":
	main()