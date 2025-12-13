# phase1/utils/tokenizer.py
class SimpleTokenizer:
    def __init__(self, texts):
        vocab = set()
        for t in texts:
            vocab.update(t.lower().split())
        self.stoi = {w: i+2 for i, w in enumerate(sorted(vocab))}
        self.stoi["<pad>"] = 0
        self.stoi["<unk>"] = 1
        self.itos = {i: w for w, i in self.stoi.items()}

    def encode(self, text, max_len):
        tokens = [self.stoi.get(w, 1) for w in text.lower().split()]
        tokens = tokens[:max_len]
        tokens += [0] * (max_len - len(tokens))
        return tokens
