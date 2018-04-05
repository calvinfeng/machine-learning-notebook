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
