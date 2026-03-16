from main_loop import load_model, get_context_words, train
from data import get_data, preprocess_data

if __name__ == "__main__":
    text = get_data()
    WORD_LIMIT = 10000
    _, _, _, bigrams = preprocess_data(text, WORD_LIMIT)

    input_weights, output_weights, word_mapping, index_mapping = load_model()

    train(10, input_weights, output_weights, word_mapping, index_mapping, bigrams)

    similar_words = get_context_words('authoritarian', input_weights, word_mapping, index_mapping)
    print(similar_words)