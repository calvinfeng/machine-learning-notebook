# Created: April, 2018
# Author(s): Calvin Feng

import numpy as np


def load_dictionary(filepath):
    with open(filepath, 'r') as file:
        text_data = file.read()
        chars = list(set(text_data))
        num_chars, num_unique_chars = len(text_data), len(chars)

        # Create a mapping from character to idx
        char_to_idx = dict()
        for i, ch in enumerate(chars):
            char_to_idx[ch] = i

        # Create a mapping from idx to character
        idx_to_char = dict()
        for i, ch in enumerate(chars):
            idx_to_char[i] = ch

        print "text document contains %d characters and has %d unique characters" % (num_chars, num_unique_chars)
        return text_data, char_to_idx, idx_to_char


def load_random_sentences(filepath, sentence_length):
    word_to_idx = {
        '<START>': 0,
        '<END>': 1,
        '<NULL>': 2
    }

    idx_to_word = {
        0: '<START>',
        1: '<END>',
        2: '<NULL>',
    }

    idx = 3
    mat = []
    with open(filepath, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row = [word_to_idx['<START>']]
            words = line.strip().split(' ')
            for word in words:
                if word_to_idx.get(word, None) is None:
                    word_to_idx[word] = idx 
                    idx_to_word[idx] = word
                    idx += 1
                row.append(word_to_idx[word])
            
            row.append(word_to_idx['<END>'])
            while len(row) < sentence_length:
                row.append(word_to_idx['<NULL>'])

            mat.append(row)
    
    return np.array(mat), word_to_idx, idx_to_word