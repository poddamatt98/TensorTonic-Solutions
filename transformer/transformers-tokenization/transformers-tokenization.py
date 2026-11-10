class SimpleTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"

    def build_vocab(self, texts: list[str]) -> None:
        """
        Builds the vocabulary in place.
        """
        
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        sorted_words = sorted(list({word.lower() for text in texts for word in text.split()}))
        for idx, word in enumerate(sorted_words):
            self.word_to_id[word] = idx+4

        self.id_to_word = {id: word for word, id in self.word_to_id.items()}
        self.vocab_size = len(self.id_to_word)


    def encode(self, text: str) -> list[int]:
        """
        Returns token IDs for the input text.
        """
        return [self.word_to_id.get(x.lower(), self.word_to_id[self.unk_token]) for x in text.split()]

    def decode(self, ids: list[int]) -> str:
        """
        Returns the decoded, space-separated text.
        """
        return ' '.join([self.id_to_word.get(id, self.unk_token) for id in ids])