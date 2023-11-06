# use bpe from sentencepiece
# use llama tokenizer config

from sentencepiece import SentencePieceProcessor

class Tokenizer:
    def __init__(self, model_path) -> None:
        self.model = SentencePieceProcessor(model_path)
        self.add_bos_token = True
        self.add_eos_token = True
        self.add_padding = False

        self.vocab_size: int = self.model.vocab_size()
        self.bos_id: int = self.model.bos_id()
        self.eos_id: int = self.model.eos_id()
        self.pad_id: int = self.model.pad_id()
        self.unk_id: int = self.model.unk_id()

    def __repr__(self):

        return (f"Tokenizer(nwords={self.n_words},"
                f"bos={self.bos_id},"
                f"eos={self.eos_id},"
                f"pad={self.pad_id},"
                f"unk={self.unk_id})")

    def encode(self, string, add_bos=True, add_eos=True, max_length=None):
        tokens = self.model.encode(string)
        if add_bos:
            tokens = [self.bos_id, *tokens]
        
        if max_length and add_eos:
            max_length = max_length - 1
        
        tokens = tokens[:max_length]

        if add_eos:
            tokens = [*tokens, self.eos_id]
        
        return tokens

    def decode(self, tokens):
        return self.model.decode(tokens)


class CharTokenizer:
    def __init__(self, path):
        self.bos = '<bos>'
        self.eos = '<eos>'

        self.ctoi = {}
        self.itoc = {}
        self.make(path, special_tokens=[self.bos, self.eos])
        self.bos_id = self.ctoi[self.bos]
        self.eos_id = self.ctoi[self.eos]
        self.vocab_size = len(self.ctoi)

    def __repr__(self):
        return (f"Tokenizer(nwords={self.vocab_size},"
                f"bos={self.bos_id},"
                f"eos={self.eos_id},"
                f"dict={self.ctoi}")

    def encode(self, string, add_bos=True, add_eos=True, max_length=None):
        ids = []
        if add_bos:
            ids = [self.bos_id]

        for char in string:
            ids.append(self.ctoi[char])

        if add_eos:
            ids.append(self.eos_id)

        return ids

    def decode(self, ids):
        return "".join(self.itoc[id] for id in ids)

    def make(self, path, special_tokens):
        ctoi = {}
        itoc = {}
        chars = set(open(path).read())
        chars.update(special_tokens)
        for idx, c in enumerate(chars):
            ctoi[c] = idx
            itoc[idx] = c

        self.ctoi = ctoi
        self.itoc = itoc
