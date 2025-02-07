#this is a deployment test file
import os
import tokeniser
import torch
import word2vec
import tt_models
import numpy as np
import pickle
# import fastapi

# app = fastapi.FastAPI()

# @app.get('/')

# async def index(option: int):

#     options = {1: "Hi there",
#             2: "Hola amigo",
#             3: "Salut amie",
#             4: "Ciao tutti",}
#     print("You selected '" + str(option) + "'")
    
#     greeting_message = options[option]
#     print(f"Option '{option}' was selected: '{greeting_message}'")
#     return {"greeting":greeting_message}



# --------------- START -----------------
script_dir = os.path.dirname(os.path.abspath(__file__))


torch.manual_seed(2)


if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device used:", device)

vocab = tokeniser.load_vocab('vocab.json')
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item()


def find_top_k_nearest_neighbors(query_embedding, tower_two_output_dict, document_dict, k=5):
    # Convert the query embedding to a tensor
    query_tensor = torch.tensor(query_embedding, dtype=torch.float32).to(device)
    
    # Convert the tower_two_output_dict to a numpy array
    embeddings = np.array(list(tower_two_output_dict.values()))

    # Prepare the output embeddings as a tensor
    output_embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Calculate cosine similarities
    similarities = torch.cosine_similarity(query_tensor.unsqueeze(0), output_embeddings)

    # Get the top k indices of the nearest neighbors
    top_k_indices = torch.topk(similarities, k=k).indices

    # Retrieve the corresponding embeddings and indices
    top_k_documents = {document_dict[idx.item()] for idx in top_k_indices}
    
    return top_k_documents, top_k_indices


t1_model = tt_models.TowerOne()
# print(t1_model)
with open("epoch2_query_tower.pt", "rb") as f:
    t1_model.load_state_dict(
        torch.load(f, map_location=device, weights_only=True)
    )

t1_model.eval()
t1_model.to(device)
query_example = "what is a furuncle boil"
query_embedding = word2vec.text_to_embeddings(query_example, vocab, 'id_vectors.npy')
query_embedding_mean = np.mean(np.array(query_embedding), axis=0)


query_tensor = torch.tensor(query_embedding_mean, dtype=torch.float32).to(device)
tower_one_output = t1_model(query_tensor)

print(f"Loading document final embeddings...")
with open('document_final_embeddings_val.pkl', 'rb') as f:
    tower_two_output_dict = pickle.load(f)

print(f"Loading document list for index...")
with open('unique_documents_val.pkl', 'rb') as f:
    unique_documents = pickle.load(f)

k=10
top_k_neighbors, top_k_indices = find_top_k_nearest_neighbors(
      query_embedding_mean
      , tower_two_output_dict
      , unique_documents
      , k)

print(f"Seach query:{query_example}")
print(f"Top {k} nearest scores:")
print(top_k_indices)
print(f"Top {k} nearest neighbors:")
print(top_k_neighbors)





# if __name__ == "__main__":
#     # Word you want to find the closest embedding for
#     option = 2

   
#     return_message = index(option) 

#     # Print the result
#     print(f"Option '{option}' was selected: '{return_message}'")
