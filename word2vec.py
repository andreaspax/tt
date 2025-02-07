import gensim.downloader as api
import numpy as np
import sklearn 
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
    
    # Get zero vector for PAD (ID 0)
    pad_vector = np.random.normal(size=300) * 0.01  # 300 = embedding dimension

    embeddings = [vectors.get(token_id, pad_vector) for token_id in token_ids]
    return embeddings

def find_similar_vectors(query_vector: np.array, all_vectors: dict, top_n: int =5) -> list[tuple]:
    # Convert to arrays
    ids = list(all_vectors.keys())
    vectors = np.array(list(all_vectors.values()))  # Shape (n, d)
    
    # Ensure query_vector is 2D (1, d)
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    elif query_vector.ndim == 2 and query_vector.shape[0] != 1:
        query_vector = np.mean(query_vector, axis=0).reshape(1, -1)
    
    # Calculate similarities
    similarities = sklearn.metrics.pairwise.cosine_similarity(query_vector, vectors)[0]
    
    # Rest of the function remains the same
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    return [(ids[i], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    # # # generate id_embeddings file id_embeddings.npy
    # # # Load the pre-trained model and generate id:vector
    model = api.load('glove-wiki-gigaword-300')

    # # # Get the vectors for vocab
    vocab = tokeniser.load_vocab('vocab.json')
    words = list(vocab.keys())

    # # # # Extract and save vectors for each word in vocab
    subset_vectors = {word: model[word] for word in words if word in model}


    # # Replace keys in loaded_vectors with their corresponding token IDs using text_to_id
    id_vectors = {vocab[word]: vector for word, vector in subset_vectors.items() if word in vocab}   
    np.save('id_vectors.npy', id_vectors)
    
    print("id_vectors.npy saved")


    # check
    test_word = "king"
    vectors = np.load('id_vectors.npy', allow_pickle=True).item()
    print(f"Checking with word {test_word}")
    embeddings = text_to_embeddings(test_word, vocab, 'id_vectors.npy')
    query_vector = np.mean(embeddings, axis=0)  # Average all token embeddings
    print(query_vector.shape)
    # similar_vectors = find_similar_vectors(query_vector, vectors, 5)
    # similar_words = [tokeniser.ids_to_text(int(vector[0]),vocab) for vector in similar_vectors]    
    # print(f"Similar words: {similar_words}")




