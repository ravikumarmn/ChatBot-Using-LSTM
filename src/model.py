import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, params, vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=params["EMBEDDING_DIM"])
        self.rnn = nn.LSTM(input_size=params["EMBEDDING_DIM"], hidden_size=params["HIDDEN_SIZE"], batch_first=True)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, params, vocab_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=params["EMBEDDING_DIM"])
        self.rnn = nn.LSTM(input_size=params["EMBEDDING_DIM"], hidden_size=params["HIDDEN_SIZE"], batch_first=True)
        self.fc = nn.Linear(in_features=params["HIDDEN_SIZE"], out_features=vocab_size)

    def forward(self, inputs, hidden, cell):
        embedded = self.embedding(inputs)
        outputs, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        logits = self.fc(outputs)
        return logits, hidden, cell

class ChatBot(nn.Module):
    def __init__(self, params, vocab_size):
        super(ChatBot, self).__init__()
        self.encoder = Encoder(params, vocab_size)
        self.decoder = Decoder(params, vocab_size)
        self.vocab_size = vocab_size
    def forward(self, inputs, targets):
        encoder_hidden, encoder_cell = self.encoder(inputs)
        decoder_inputs = targets[:, :-1]
        decoder_targets = targets[:, 1:]
        logits, _, _ = self.decoder(decoder_inputs, encoder_hidden, encoder_cell)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), decoder_targets.flatten())
        return logits, loss
