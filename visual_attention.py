from rnn import *



def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	maxSentenceLength = 31
	embedding_dim = 200
	epochs = 15
	hidden_dim = 250
	dropout = 0
	weight_decay = 0.0001
	lr_decay = 0.95
	decay_start = 5
	patience = 5
	output_dim = len(emotions)
	batch_size = 256
	threshold = 0.5
	attention = True
	r = 1
	lr = 1e-2
	init_lr = lr
	filename = "rnn"

	print("Loading embeddings..\n")
	rawEmbedding = GloVe(name='twitter.27B', dim=embedding_dim)
	#rawEmbedding = FastText(language='en')

	#get data
	train = getTrainSet()

	#clean text
	train.text = train.text.apply(cleanTextForEmbedding)
	train = train[train.text.str.len() > 0]

	text_field = Field(
		sequential=True,
		tokenize='basic_english', 
		fix_length=maxSentenceLength,
		lower=True,
		include_lengths=True,
	)

	#build vocab
	preprocessed_text = train.text.apply(
		lambda x: text_field.preprocess(x)
	)

	text_field.build_vocab(
		preprocessed_text, 
		vectors=rawEmbedding,
	)

	vocab = text_field.vocab

	dataset = makeDataset(train, emotions, text_field)

	model = BiLSTM(vocab.vectors.to(device), embedding_dim, hidden_dim, output_dim, maxSentenceLength, device, n_layers=1, r=r, dropout=dropout, attention=attention)
	model.to(device)
	checkpoint = torch.load("rnn.pt")
	model.load_state_dict(checkpoint['model_state_dict'])

	trainloader = Iterator(dataset, 1)

	#get one example
	example = next(iter(trainloader))
	print(train.text[0])
	data, length = example.text

	output, h, att_weights = model(data.to(device), length)
	att_weights = att_weights.cpu()
	print(att_weights)
	print(att_weights.size())






if __name__ == "__main__":
	main()