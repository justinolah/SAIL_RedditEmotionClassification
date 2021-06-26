import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')
from helpers import *
from learn import *

MAX_SENTENCE_LENGTH = 33

def getFasttextModel():
    print("Loading pretrained fasttext model...")
    ft = fasttext.load_model('cc.en.300.bin')
    wordVecLength = 300
    print("Model loaded\n")
    return ft, wordVecLength

def cleanTextForEmbedding(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    words = text.split()
    return " ".join([word for word in words if len(word) >= MIN_WORD_LENGTH])

def getSentenceVectorAverageWord(text, ft):
    return ft.get_sentence_vector(text)

def getSentenceVectorPadded(text, ft, maxLength, wordVecLength):
    words = text.split()
    vec = []
    word_count = 0
    for word in words:
        if word_count == maxLength:
            break
        vec += list(ft.get_word_vector(word))
        word_count += 1

    while word_count < maxLength:
        vec += list([0] * wordVecLength)
        word_count += 1

    assert len(vec) == maxLength * wordVecLength

    return vec

def trainSupervisedFasttext(train, test, emotions):

    #format labels

    fasttext_params = {
        'input': train.text,
        'lr': 0.1,
        'lrUpdateRate': 1000,
        'thread': 8,
        'epoch': 10,
        'wordNgrams': 1,
        'dim': 100,
        'loss': 'ova'
    }

    model = fasttext.train_supervised(**fasttext_params)

    print('vocab size: ', len(model.words))
    print('label size: ', len(model.labels))

    yhat = model.predict(test.text)

def main():
    emotions = getEmotions()
    emotions.remove("neutral")

    train = getTrainSet()
    test = getTestSet()
    #val = getValSet()

    train.text = train.text.apply(cleanTextForEmbedding)
    test.text = test.text.apply(cleanTextForEmbedding)

    print("Loading pretrained fasttext model...")
    ft = fasttext.load_model('cc.en.300.bin')
    print("Model loaded")
    print("Corpus lenght:", len(ft.words))
    wordVecLength = len(ft.get_word_vector('test'))
    print("Word vec length:", wordVecLength)
    print("")

    #x_train = train_text.apply(lambda x: getSentenceAverageWord(x,ft))
    #x_test = test_text.apply(lambda x: getSentenceAverageWord(x,ft))

    x_train = train.text.apply(lambda x: getSentenceVectorPadded(x, ft, MAX_SENTENCE_LENGTH, wordVecLength))
    x_test = test.text.apply(lambda x: getSentenceVectorPadded(x, ft, MAX_SENTENCE_LENGTH, wordVecLength))
    print(x_train)
    print(len(x_train[0]))
    return

    #todo add labels to text
    #todo write to a file first
    trainSupervisedFasttext(train, test, emotions)

if __name__ == "__main__":
    main()