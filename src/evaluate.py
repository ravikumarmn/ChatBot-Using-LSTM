import json
import re
import unicodedata

import torch
import torch.nn as nn

from model import ChatBot, Decoder, Encoder


class SearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, start_token, end_token, vocab):
        super(SearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.start_token = start_token
        self.end_token = end_token
        self.word2index = vocab
        self.index2word = {v: k for k, v in self.word2index.items()}

    def forward(self, input_sequence, max_length=100):
        hidden_state, encoder_cell = self.encoder(input_sequence)
        decoder_input = torch.ones((1, 1), dtype=torch.long) * self.start_token
        all_tokens = torch.zeros((0), dtype=torch.long)
        all_scores = torch.zeros((0))
        
        for _ in range(max_length):
            logits, _, _ = self.decoder(decoder_input, hidden_state, encoder_cell)
            decoder_scores, decoder_input = torch.max(logits, dim=-1)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=1)
            all_scores = torch.cat((all_scores, decoder_scores), dim=1)
            
            if decoder_input.item() == self.end_token:
                break

        return all_tokens, all_scores

def evaluate_fn(searcher, user_input, word2index):
    index2word = {v: k for k, v in word2index.items()}
    indexes_batch = [word2index[word] for word in user_input.split(' ')] + [word2index["<end>"]]

    input_batch = torch.LongTensor(indexes_batch)
    tokens, scores = searcher(input_batch.view(1, -1))

    decoded_words = [index2word[token.item()] for token in tokens.flatten()]
    decoded_words = [x for x in decoded_words if not (x == '<end>' or x == '<pad>')]
    return decoded_words

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def evaluateInput(searcher, vocab):
    while True:
        try:
            input_sentence = input('> ')
            if input_sentence == 'q' or input_sentence in ['quit','exit','bye','Exit','Bye','Quit']:
                break
            input_sentence = normalizeString(input_sentence)
            output_words = evaluate_fn(searcher, input_sentence, vocab)
            print('Bot:', ' '.join(output_words))
        except KeyError:
            print("Sorry! I didn't get you")
def main():
    vocab = json.load(open("dataset/vocab.json","r"))
    vocab_size = len(vocab)
    checkpoints = torch.load(open("checkpoints/lstm/lstm_batch_size_64_hidden_size128.pt","rb"))
    params = checkpoints["params"]

    encoder = Encoder(params, vocab_size)
    decoder = Decoder(params, vocab_size)
    model = ChatBot(params, vocab_size)

    encoder.load_state_dict(checkpoints['encoder'])
    decoder.load_state_dict(checkpoints['decoder'])
    model.load_state_dict(checkpoints["model_state_dict"])
    model.eval()
    searcher = SearchDecoder(encoder, decoder,vocab["<start>"],vocab['<end>'],vocab)
    evaluateInput(searcher,vocab)

if __name__=="__main__":
    main()
