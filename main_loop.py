import numpy as np
from data import get_data, preprocess_data


text = get_data()
words_dict, word_mapping, bigrams = preprocess_data(text)





