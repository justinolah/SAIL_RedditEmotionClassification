from nltk.corpus import wordnet as wn
from helpers import *

emotions = getEmotions()
emotions.remove("neutral")

def getHypernyms(synsets):
	hypernyms = []
	for synset in synsets:
		h = synset.hypernyms()
		if len(h) > 0:
			hypernyms.append(*h)
	return list(set(hypernyms))

def getNames(synsets):
	names = []
	for synset in synsets:
		names.append(synset.name().split(".")[0])
	return list(set(names))

def getDefinition(emotion):
	return wn.synsets(emotion)[0].definition()

def main():
	for emotion in emotions:
		print(emotion)
		synsets = wn.synsets(emotion)#, wn.NOUN)
		synonyms = getNames(synsets)
		print(synsets[0].definition())
		print(synonyms)
		hypernyms = getHypernyms(synsets)
		hypernyms = getNames(hypernyms)
		print(hypernyms)
		print("")

if __name__ == "__main__":
	main()