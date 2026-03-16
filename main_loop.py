import numpy as np
from data import get_data, preprocess_data

EMBEDDING_SIZE = 50

def save_model(input_weights, output_weights, word_mapping, index_mapping, filename="model.npz"):
    np.savez(filename, w1=input_weights, w2=output_weights, mapping=word_mapping, i_mapping=index_mapping)

def load_model(filename="model.npz"):
    data = np.load(filename, allow_pickle=True)
    w1 = data['w1']
    w2 = data['w2']
    mapping = data['mapping'].item()
    index_mapping = data['i_mapping'].item()
    return w1, w2, mapping, index_mapping

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def cosine_similarity(a, b):
    return (a @ b) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_context_words(word, embeddings, word_mapping, index_mapping):
    word_index = word_mapping[word]
    similarities = []
    context_words = []

    for index, embedding in enumerate(embeddings):
        if index != word_index:
            similarity = cosine_similarity(embeddings[word_index], embedding)
            similarities.append((index, similarity))
    similarities.sort(key=lambda x: x[1], reverse=True)
    for i in range(5):
        context_word = index_mapping[similarities[i][0]]
        context_words.append(context_word)

    return context_words

def train(epochs, input_weights, output_weights, word_mapping, index_mapping, bigrams, lr=0.01):

    word_size = len(word_mapping)

    for epoch in range(epochs):
        total_loss = 0
        for target_word, context_word in bigrams:
            if target_word not in word_mapping or context_word not in word_mapping:
                continue    

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
        if(epoch % 5 == 0):
            save_model(input_weights, output_weights, word_mapping, index_mapping)
    save_model(input_weights, output_weights, word_mapping, index_mapping)
        


