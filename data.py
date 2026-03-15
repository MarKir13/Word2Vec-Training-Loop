import kagglehub
import os

def get_data():
    path = kagglehub.dataset_download("yorkyong/text8-zip")
    path = os.path.join(path, "text8")

    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    
    return text

def preprocess_data(data, word_limit):

    words = data.split()
    words = words[:word_limit]
    words_dict = set(words)
    word_mapping = {}

    for i, word in enumerate(words_dict):
        word_mapping[word] = i

    bigrams = []

    for i in range(len(words) - 1):
        bigrams.append((words[i], words[i+1]))
        bigrams.append((words[i+1], words[i]))
    return words_dict, word_mapping, bigrams
