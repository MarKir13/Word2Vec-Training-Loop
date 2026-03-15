import numpy as np
from data import get_data, preprocess_data

EMBEDDING_SIZE = 50
WORD_LIMIT = 20000

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

text = get_data()
words_dict, word_mapping, bigrams = preprocess_data(text, WORD_LIMIT)

word_size = len(words_dict)

input_weights = np.random.rand(word_size, EMBEDDING_SIZE) - 0.5
output_weights = np.random.rand(EMBEDDING_SIZE, word_size) - 0.5

epochs = 100
lr = 0.001

for epoch in range(epochs):
    total_loss = 0
    for target_word, context_word in bigrams:
        h = input_weights[word_mapping[target_word], :]

        u = h @ output_weights
        y_pred = softmax(u)

        y_true = np.zeros((word_size))
        y_true[word_mapping[context_word]] = 1

        error = y_pred.copy() - y_true 
        total_loss -= np.log(y_pred[word_mapping[context_word]])

        output_grad = np.outer(h, error)
        backward_grad = error @ output_weights.T

        output_weights -= lr * output_grad
        input_weights[word_mapping[target_word], :] -= lr * backward_grad
        
    print(f"Epoch: {epoch} | Loss: {total_loss / len(bigrams)}")
        
        


