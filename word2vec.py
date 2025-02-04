import gensim.downloader as api
import numpy as np
import json
import tokeniser

def text_to_embeddings(text: str, vocab: dict, vector_file: np.array) -> list[np.array]:
    """
    Convert a text string into an embedding using pre-trained word vectors.
    
    Args:
        text (str): The input text string to be embedded.
        vocab (dict): The vocabulary mapping.
        vectors (np.array): The pre-trained word vectors."""
    vectors = np.load(vector_file, allow_pickle=True).item()
    token_ids = tokeniser.text_to_ids(text, vocab)
    embeddings = [vectors[token_id] for token_id in token_ids]
    return embeddings


if __name__ == "__main__":
    # # generate id_embeddings file id_embeddings.npy
    # # Load the pre-trained model and generate id:vector
    # model = api.load('glove-wiki-gigaword-300')

    # # Get the vector for a word
    # vector = model['king']  # Returns a 300-dimensional numpy array
    # print("Vector for 'king':", vector)
    # print("Vector shape:", vector.shape) 

    # # similar words
    # similar_words = model.most_similar('king', topn=5)
    # print("Words similar to 'king':", similar_words)


    # # # Words of interest
    # with open('vocab.json', 'r') as f:
    #     vocab = json.load(f)
    # print("vocab")
    # print(len(vocab))
    # # # words = list(vocab.keys())


    # words = ['king', 'queen', 'man', 'woman', 'country']

    # # # Extract and save vectors
    # subset_vectors = {word: model[word] for word in words if word in model}
    # np.save('sample_subset_vectors.npy', subset_vectors)

    # print(subset_vectors)

    # Load the saved sample_subset_vectors.npy
    loaded_vectors = np.load('sample_subset_vectors.npy', allow_pickle=True).item()
    # print("Loaded subset vectors:", loaded_vectors)

    print(len(loaded_vectors))
    print(loaded_vectors.keys())
    print(np.array(list(loaded_vectors.values())).shape)

    example = "This king and queen are a Man Woman country?"

    vocab = tokeniser.load_vocab('vocab.json')
    print("the word 'king' has an id of", vocab["king"])


    reverse_vocab = {idx: word for word, idx in vocab.items()}
    print("id 48576 is the word", reverse_vocab[48576])
    # # Create a reverse vocabulary mapping for easy lookup
    # print(dict(list(reverse_vocab.items())[:10]))  # Show first 10 items in vocab


    # Replace keys in loaded_vectors with their corresponding IDs
    id_vectors = {reverse_vocab[word]: vector for word, vector in loaded_vectors.items() if word in reverse_vocab}

    # Optionally, save the new id_vectors to a file
    np.save('id_vectors.npy', id_vectors)
    
    print(id_vectors.keys())
    # print("ID Vectors:", id_vectors)

    check = list(id_vectors.keys())
    print(check)
    # print(tokeniser.ids_to_text([48574, 70996], vocab))
    print(reverse_vocab[48574])


    # # Create a reverse vocabulary mapping for easy lookup
    # reverse_vocab = {word: idx for idx, word in enumerate(vocab.keys())}

    # # Replace keys in loaded_vectors with their corresponding IDs
    # id_vectors = {reverse_vocab[word]: vector for word, vector in loaded_vectors.items() if word in reverse_vocab}

    # # Optionally, save the new id_vectors to a file
    # np.save('id_vectors.npy', id_vectors)

    # print("ID Vectors:", id_vectors)



