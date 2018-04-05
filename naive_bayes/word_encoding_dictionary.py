from sortedcontainers import SortedSet


class WordEncodingDictionary:
    """Word encoding dictionary keeps a bi-directional map between numeric codes and words they stand for.
    """
    def __init__(self):
        self.word_to_code_dict = {}
        self.code_to_word_dict = {}

    def word_to_code(self, word):
        if word not in self.word_to_code_dict:
            code = len(self.word_to_code_dict)
            self.word_to_code_dict[word] = code
            self.code_to_word_dict[code] = word

        return self.word_to_code_dict[word]

    def code_to_word(self, code):
        if code not in self.code_to_word_dict:
            raise "Code %s not recorded!" % code

        return self.code_to_word_dict[code]

    def encode_text(self, text):
        codes = SortedSet()
        for word in text.split():
            codes.add(self.word_to_code(word))
        return codes

    def decode_codes_set(self, codes):
        cls = type(codes)
        return cls(map(self.code_to_word, codes))
