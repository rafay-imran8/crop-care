class DecoderTokenizer:
    def __init__(self, text):
        self.sos = "<sos>"
        self.eos = "<eos>"

        words = text.split()
        vocab = sorted(set(words))

        self.stoi = {self.sos: 0, self.eos: 1}
        self.stoi.update({w: i + 2 for i, w in enumerate(vocab)})
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[self.sos]] + \
               [self.stoi[w] for w in text.split()] + \
               [self.stoi[self.eos]]

    def decode(self, tokens):
        return " ".join(self.itos[t] for t in tokens if t > 1)