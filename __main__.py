from main_loop import load_model, get_context_words, train
from data import get_data, preprocess_data

if __name__ == "__main__":
    text = get_data()
    WORD_LIMIT = 10000
    _, _, _, bigrams = preprocess_data(text, WORD_LIMIT)

    input_weights, output_weights, word_mapping, index_mapping = load_model()

    train(50, input_weights, output_weights, word_mapping, index_mapping, bigrams)

    word = 'authoritarian'
    similar_words = get_context_words(word, input_weights, word_mapping, index_mapping)
    print(f"Five best context words for word {word}: {similar_words}")