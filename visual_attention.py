from rnn import *

class BiLSTM(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, maxlen, device, n_layers=1, r=1, dropout=0, attention=False):
		super(BiLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device
		self.maxlen = maxlen
		self.attention = attention
		self.r = r

		self.tanh = torch.tanh

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=False)
		if attention:
			self.W_s1 = nn.Linear(hidden_dim * 2, 350)
			self.W_s2 = nn.Linear(350, self.r)
			self.cat_layer = nn.Linear(self.r * 2 * hidden_dim, 1000)
			self.fc = nn.Linear(1000, output_dim)
		else:
			self.fc = nn.Linear(hidden_dim * 2, output_dim)

		self.dropout = nn.Dropout(dropout)

	def attention_net(self, lstm_out, batch_size):
		lstm_out = lstm_out.permute(1,0,2)
		att_weights = self.W_s2(self.tanh(self.W_s1(lstm_out)))
		att_weights = att_weights.permute(0, 2, 1)
		att_weights = F.softmax(att_weights, dim=2)

		hidden_matrix = torch.bmm(att_weights, lstm_out)

		out = self.cat_layer(hidden_matrix.view(batch_size, -1))

		return out, att_weights

	def forward(self, x, lengths):
		batch_size = x.size(1)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)

		hidden = self.initHidden(batch_size)
	
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out, lengths = pad_packed_sequence(lstm_out, total_length=self.maxlen)

		if self.attention:
			out, att_weights = self.attention_net(lstm_out, batch_size)
		else:
			out = torch.zeros_like(lstm_out[0])
			for i, length in enumerate(lengths):
				out[i] = lstm_out[length-1,i]
		
		out = self.dropout(out)
		out = self.fc(out)

		return out, hidden, att_weights

	def initHidden(self, batch_size):
		return (torch.zeros((2*self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device),
			torch.zeros((2*self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device))


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

	for i in range(5):
		print("Example:", i)
		example = next(iter(trainloader))
		sentence = train.text[i]
		words = sentence.split()
		print(sentence)
		data, length = example.text

		output, h, att_weights = model(data.to(device), length)
		att_weights = att_weights.cpu()

		soft = nn.Softmax(dim=0)

		heat = soft(att_weights.squeeze()[0:len(words)])
		print(heat)







if __name__ == "__main__":
	main()