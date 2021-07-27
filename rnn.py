from pytorch_model import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Attention(nn.Module):
	def __init__(self, feature_dim, step_dim, **kwargs):
		super(Attention, self).__init__(**kwargs)
		
		self.supports_masking = True #delete?

		self.feature_dim = feature_dim
		self.step_dim = step_dim
		self.features_dim = 0 #delete?
		
		weight = torch.zeros(feature_dim, 1)
		nn.init.kaiming_uniform_(weight)
		self.weight = nn.Parameter(weight)
		
		
	def forward(self, x, mask=None):
		feature_dim = self.feature_dim 
		step_dim = self.step_dim

		print(self.step_dim)
		print(x)
		print(x.size()) # x -> [24, 256, 250]
		print(self.weight)
		print(self.weight.size()) # weight -> [250 x 1]
		eij = x.matmul(self.weight)
		print(eij)
		print(eij.size())
		pizza

		eij = torch.mm(
			x.contiguous().view(-1, feature_dim), 
			self.weight
		).view(-1, step_dim)
			
		eij = torch.tanh(eij)
		a = torch.exp(eij)
		
		if mask is not None:
			a = a * mask

		a = a / (torch.sum(a, 1, keepdim=True) + 1e-10) #softmax

		weighted_input = x * torch.unsqueeze(a, -1)
		return torch.sum(weighted_input, 1)

class UniLSTM(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim, maxlen, device, n_layers=1, dropout=0, attention=True):
		super(UniLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device
		self.maxlen = maxlen
		self.attention = attention

		self.tanh = torch.tanh

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=False)
		if attention:
			#self.attention_layer = Attention(hidden_dim, maxlen)
			self.W_s1 = nn.Linear(hidden_dim, 350)
			self.W_s2 = nn.Linear(350, 1)
			self.cat_layer = nn.Linear(1*hidden_dim, 1000)
			self.fc = nn.Linear(1000, output_dim)
		else:
			self.fc = nn.Linear(hidden_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def attention_net(self, lstm_out, batch_size):
		lstm_out = lstm_out.permute(1,0,2)
		att_weights = self.W_s2(self.tanh(self.W_s1(lstm_out)))
		att_weights = att_weights.permute(0, 2, 1)
		att_weights = F.softmax(att_weights, dim=2)

		hidden_matrix = torch.bmm(att_weights, lstm_out)

		out = self.cat_layer(hidden_matrix.view(batch_size, -1))

		return out

	def forward(self, x, lengths):
		batch_size = x.size(1)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)

		hidden = self.initHidden(batch_size)
	
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out, lengths = pad_packed_sequence(lstm_out,total_length=self.maxlen)

		if self.attention:
			out = self.attention_net(lstm_out, batch_size)#self.attention_layer(lstm_out)
		else:
			out = torch.zeros_like(lstm_out[0])
			for i, length in enumerate(lengths):
				out[i] = lstm_out[length-1,i]
		
		out = self.dropout(out)
		out = self.fc(out)

		return out, hidden

	def initHidden(self, batch_size):
		return (torch.zeros((self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device),
			torch.zeros((self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device))

class UniGRU(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim,  maxlen, device, n_layers=1, dropout=0):
		super(UniGRU, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device

		self.gru = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=False)
		self.fc = nn.Linear(hidden_dim, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths):
		batch_size = x.size(1)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)

		hidden = self.initHidden(batch_size)
	
		gru_out, hidden = self.gru(embeds, hidden)
		gru_out, lengths = pad_packed_sequence(gru_out)

		out = torch.zeros_like(gru_out[0])

		for i, length in enumerate(lengths):
			out[i] = gru_out[length-1,i]
		
		out = self.dropout(out)
		out = self.fc(out)

		return out, hidden

	def initHidden(self, batch_size):
		return torch.zeros((self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device)

class BiLSTM(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim,  maxlen, device, n_layers=1, dropout=0):
		super(BiLSTM, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device

		self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=False)
		self.fc = nn.Linear(hidden_dim * 2, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths):
		batch_size = x.size(1)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)

		hidden = self.initHidden(batch_size)
	
		lstm_out, hidden = self.lstm(embeds, hidden)
		lstm_out, lengths = pad_packed_sequence(lstm_out)

		out = torch.zeros_like(lstm_out[0])

		for i, length in enumerate(lengths):
			out[i] = lstm_out[length-1,i]

		#h_f = out[:,:,self.hidden_dim:]
		#h_r = out[:,:, :self.hidden_dim]
		
		out = self.dropout(out)
		out = self.fc(out)

		return out, hidden

	def initHidden(self, batch_size):
		return (torch.zeros((2*self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device),
			torch.zeros((2*self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device))

class BiGRU(nn.Module):
	def __init__(self, embedding, embedding_dim, hidden_dim, output_dim,  maxlen, device, n_layers=1, dropout=0):
		super(BiGRU, self).__init__()
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		self.embedding = embedding
		self.n_layers = n_layers
		self.device = device

		self.gru= nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout, bidirectional=True, batch_first=False)
		self.fc = nn.Linear(hidden_dim * 2, output_dim)

		self.dropout = nn.Dropout(dropout)

	def forward(self, x, lengths):
		batch_size = x.size(1)
		embeds = self.embedding[x]
		embeds = pack_padded_sequence(embeds, lengths, enforce_sorted=False)

		hidden = self.initHidden(batch_size)
	
		gru_out, hidden = self.gru(embeds, hidden)
		gru_out, lengths = pad_packed_sequence(gru_out)

		out = torch.zeros_like(gru_out[0])

		for i, length in enumerate(lengths):
			out[i] = gru_out[length-1,i]
		
		out = self.dropout(out)
		out = self.fc(out)

		return out, hidden

	def initHidden(self, batch_size):
		return torch.zeros((2*self.n_layers, batch_size, self.hidden_dim), requires_grad=True).to(self.device)

def trainNN(model, trainloader, batch_size, devData, devLengths, devLabels, optimizer, loss_fn, threshold, device):
	counter = 0
	train_running_loss = 0.0
	sigmoid = nn.Sigmoid()

	model.train()

	allTargets = []
	allPredictions = []

	for i, batch in enumerate(trainloader):
		counter += 1
		inputs, lengths = batch.text
		labels = batch.labels.float()
		allTargets.append(labels.detach())

		inputs, labels = inputs.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs, h = model(inputs, lengths)
		outputs = outputs.squeeze()
		allPredictions.append((outputs.cpu() > threshold).int().detach())

		loss = loss_fn(outputs, labels)
		loss.backward()
		optimizer.step()

		train_running_loss += loss.item()

	trainLoss = train_running_loss / counter

	model.eval()
	devOutput, val_h = model(devData.to(device), devLengths)
	devOutput = devOutput.squeeze()
	devLoss = loss_fn(devOutput, devLabels.to(device)).item()
	devOutput = sigmoid(devOutput)

	allPredictions = np.concatenate(allPredictions)
	allTargets = np.concatenate(allTargets)

	f1_train = f1_score(allTargets, allPredictions, average='macro')
	f1_dev = f1_score(devLabels.detach(), (devOutput.cpu() > threshold).int().detach(), average='macro')

	return trainLoss, devLoss, f1_train, f1_dev

def main():
	emotions = getEmotions()
	emotions.remove("neutral")

	if torch.cuda.is_available():
		print("Using cuda...")
		device = torch.device("cuda")
	else:
		print("Using cpu...")
		device = torch.device("cpu")

	#parameters
	maxSentenceLength = 31
	embedding_dim = 200
	epochs = 7
	hidden_dim = 250
	dropout = 0
	weight_decay = 0.0001
	lr_decay = 0.95
	output_dim = len(emotions)
	batch_size = 256
	threshold = 0.5
	lr = 1e-2
	init_lr = lr
	filename = "rnn"

	print("Loading embeddings..\n")
	rawEmbedding = GloVe(name='twitter.27B', dim=embedding_dim)
	#rawEmbedding = FastText(language='en')

	#get data
	train = getTrainSet()
	test = getTestSet()
	dev = getValSet()

	#clean text
	train.text = train.text.apply(cleanTextForEmbedding)
	test.text= test.text.apply(cleanTextForEmbedding)
	dev.text = dev.text.apply(cleanTextForEmbedding)

	#remove empty sentences
	train = train[train.text.str.len() > 0]
	test = test[test.text.str.len() > 0]
	dev = dev[dev.text.str.len() > 0]

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

	#print("Max sentence length:", np.array([len(row) for row in preprocessed_text.to_list()]).max()) #gets max sentence length

	text_field.build_vocab(
		preprocessed_text, 
		vectors=rawEmbedding,
	)

	vocab = text_field.vocab

	#Make datasets
	dataset = makeDataset(train, emotions, text_field)
	testset = makeDataset(test, emotions, text_field)
	devset = makeDataset(dev, emotions, text_field)

	#pytorch model
	print("Training NN...")
	torch.manual_seed(42)
	rnn = UniLSTM(vocab.vectors.to(device), embedding_dim, hidden_dim, output_dim, maxSentenceLength, device, n_layers=1, dropout=dropout, attention=True)
	rnn.to(device)

	optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)

	loss_fn= nn.BCEWithLogitsLoss(pos_weight= 8*torch.ones(len(emotions)).to(device))
	#loss_fn = wlsep

	#trainloader = BucketIterator(dataset, batch_size, sort_key=lambda x: len(x.text), repeat=True, shuffle=True, sort_within_batch=True) #runs too slow for some reason
	trainloader = Iterator(dataset, batch_size)

	#train model
	rnn.train()
	trainLoss = []
	devLoss = []
	trainF1 = []
	devF1 = []
	devbatch = next(iter(Iterator(devset, len(devset))))
	devData, devLengths = devbatch.text
	devLabels = devbatch.labels.float()

	for epoch in range(epochs):
		print("Epoch:", epoch)
		epoch_loss, dev_loss, f1_train, f1_dev = trainNN(rnn, trainloader, batch_size, devData, devLengths, devLabels, optimizer, loss_fn, threshold, device)
		trainLoss.append(epoch_loss)
		devLoss.append(dev_loss)
		trainF1.append(f1_train)
		devF1.append(f1_dev)
		print("Training Loss:", epoch_loss)
		print("Dev Loss:", dev_loss)
		print("Training Macro F1:", f1_train)
		print("Dev Macro F1:", f1_dev, "\n")

		lr *= lr_decay
		optimizer = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=weight_decay)

	print("Training complete\n")

	bestEpochDevF1 = np.argmax(np.array(devF1))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")

	#learning curve 
	fig, (ax1, ax2) = plt.subplots(2, figsize=(11, 9))
	ax1.plot(trainLoss, color='b', label='Training loss')
	ax1.plot(devLoss, color='r', label='Dev loss')
	fig.suptitle(f"Hidden:{hidden_dim}, LR:{init_lr}, BS:{batch_size}, Embedding Size: {embedding_dim}, LR Decay: {lr_decay}, Weight Decay: {weight_decay}")
	ax1.set(xlabel='Epochs', ylabel="Loss")
	ax1.legend()
	ax2.plot(trainF1, color='b', label='Training Macro F1')
	ax2.plot(devF1, color='r', label='Dev Macro F1')
	ax2.set(xlabel='Epochs', ylabel="Macro F1")
	ax2.legend()
	fig.savefig('plots/learningcurve_rnn.png')

	#Testing metrics
	rnn.eval()
	sigmoid = nn.Sigmoid()
	testbatch = next(iter(Iterator(testset, len(devset))))
	data, lengths = testbatch.text
	labels = testbatch.labels.float()

	outputs, h = rnn(data.to(device), lengths)
	outputs = sigmoid(outputs.squeeze())
	prediction = (outputs > threshold).int().cpu()

	accuracy = accuracy_score(labels, prediction)
	print("Subset Accuracy:", accuracy)
	print(classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=False))
	print("Best Dev F1:", devF1[bestEpochDevF1], "at epoch", bestEpochDevF1, "\n")
	report = classification_report(labels, prediction, target_names=emotions, zero_division=0, output_dict=True)

	#export resuls to csv
	micro = list(report['micro avg'].values())
	micro.pop() 
	macro = list(report['macro avg'].values())
	macro.pop()
	scores = [accuracy, *micro, *macro]
	results = pd.DataFrame(data=[scores], columns=['accuracy', 'micro_precision', 'micro_recall', 'micro_f1', 'macro_precision', 'macro_recall', 'macro_f1'])
	results.to_csv("tables/" + filename + "_results.csv")

	#confusion matrix
	multilabel_confusion_matrix(np.array(labels), outputs.cpu().detach().numpy(), emotions, top_x=3, filename=filename)

if __name__ == "__main__":
	main()