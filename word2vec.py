import gensim.downloader as api
import numpy as np
# import sklearn 
import json
import tokeniser
import re

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

# def find_similar_vectors(query_vector: np.array, all_vectors: dict, top_n: int =5) -> list[tuple]:
#     """
#     Find most similar vectors using cosine similarity
    
#     Args:
#         query_vector: Shape (d,) - embedding to compare against
#         all_vectors: Dict {id: vector} of candidate vectors
#         top_n: Number of results to return
#     """
#     # Convert to arrays
#     ids = list(all_vectors.keys())
#     vectors = np.array(list(all_vectors.values()))  # Shape (n, d)
    
#     # Calculate similarities
#     similarities = sklearn.metrics.pairwise.cosine_similarity([query_vector], vectors)[0]
    
#     # Get top matches
#     top_indices = np.argsort(similarities)[-top_n:][::-1]
#     return [(ids[i], similarities[i]) for i in top_indices]

if __name__ == "__main__":
    # # generate id_embeddings file id_embeddings.npy
    # # Load the pre-trained model and generate id:vector
    model = api.load('glove-wiki-gigaword-300')

    # # Get the vectors for vocab
    vocab = tokeniser.load_vocab('vocab.json')
    words = list(vocab.keys())

    # # # Extract and save vectors for each word in vocab
    subset_vectors = {word: model[word] for word in words if word in model}
    # np.save('subset_vectors.npy', subset_vectors) -- to save the vectors if wanted

    # Replace keys in loaded_vectors with their corresponding token IDs using text_to_id
    id_vectors = {vocab[word]: vector for word, vector in subset_vectors.items() if word in vocab}   
    np.save('id_vectors.npy', id_vectors)
    print("id_vectors.npy saved")
    print(len(id_vectors))
    print(np.array(list(id_vectors.values())).shape)





