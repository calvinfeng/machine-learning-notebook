import os
import os.path
import io

FILEPATH = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.join(FILEPATH, "datasets")


class Email(object):
    def __init__(self, path, content, word_encoding_dictionary, label):
        self.path = path
        self.codes = word_encoding_dictionary.encode_text(content)
        self.label = label
        self.word_encoding_dictionary = word_encoding_dictionary

    def text_content(self):
        return type(self).read_text_content(self.path)

    def word_set(self):
        return self.word_encoding_dictionary.decode_code_set(self.codes)

    @classmethod
    def read(cls, path, word_encoding_dictionary, label):
        return Email(
            path=path,
            content=cls.read_text_content(path),
            word_encoding_dictionary=word_encoding_dictionary,
            label=label
        )

    @classmethod
    def read_text_content(cls, path):
        full_path = os.path.join(DATA_DIR, path)
        with io.open(full_path, mode="r", encoding="iso-8859-1") as f:
            try:
                return f.read()
            except:
                print "Error with %s" % path
                raise
