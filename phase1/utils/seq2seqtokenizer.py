class Seq2SeqTokenizer:
    def __init__(self, texts):
        self.pad = "<pad>"
        self.sos = "<sos>"
        self.eos = "<eos>"

        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())

        # Indices: 0=pad, 1=sos, 2=eos, 3+=words
        self.stoi = {self.pad: 0, self.sos: 1, self.eos: 2}
        self.stoi.update({w: i + 3 for i, w in enumerate(sorted(vocab))})
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text):
        return [self.stoi[self.sos]] + [self.stoi[w] for w in text.lower().split()] + [self.stoi[self.eos]]

    def decode(self, tokens):
        return " ".join([self.itos[t] for t in tokens if t > 2])