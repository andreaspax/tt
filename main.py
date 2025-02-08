#this is a deployment test file
import os
import tokeniser
import torch
import word2vec
import tt_models
import numpy as np
import pickle
import fastapi

app = fastapi.FastAPI()

k=5

if torch.cuda.is_available():
            device = torch.device("cuda")
elif torch.backends.mps.is_available():
            device = torch.device("mps")
else:
            device = torch.device("cpu")
print("device used:", device)

vocab = tokeniser.load_vocab('vocab.json')
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item()

t1_model = tt_models.TowerOne()

with open("epoch2_query_tower.pt", "rb") as f:
        t1_model.load_state_dict(
            torch.load(f, map_location=device, weights_only=True)
        )

t1_model.eval()
t1_model.to(device)


print(f"Loading document final embeddings...")
with open('document_final_embeddings_val.pkl', 'rb') as f:
        tower_two_output_dict = pickle.load(f)

print(f"Loading document list for index...")
with open('unique_documents_val.pkl', 'rb') as f:
        unique_documents = pickle.load(f)

def find_top_k_nearest_neighbors(query_tensor, tower_two_output_dict, document_dict, k=5):
    
    # Convert the tower_two_output_dict to a numpy array
    embeddings = np.array(list(tower_two_output_dict.values()))

    # Prepare the output embeddings as a tensor
    output_embeddings = torch.tensor(embeddings, dtype=torch.float32).to(device)

    # Calculate cosine similarities
    similarities = torch.cosine_similarity(query_tensor.unsqueeze(0), output_embeddings)

    # Get the top k indices of the nearest neighbors
    top_k_index = torch.topk(similarities, k=k).indices

    top_k_scores = torch.topk(similarities, k=k).values


    # Retrieve the corresponding embeddings and indices
    top_k_documents = {document_dict[idx.item()] for idx in top_k_index}
    
    return top_k_documents, top_k_scores

@app.get('/')
async def root():
    return {"message": "Hello World"}

@app.get("/search")
async def search(query: str):
    # if query.strip()==  '': return []
# if __name__ == "__main__":
    # query = "what is a furuncle boil"
    query_embedding = word2vec.text_to_embeddings(query, vocab, 'id_vectors.npy')
    if query_embedding is None: return [{"no_embedding": True}]
    query_embedding_mean = np.mean(np.array(query_embedding), axis=0)
    query_tensor = torch.tensor(query_embedding_mean, dtype=torch.float32).to(device)
    tower_one_output = t1_model(query_tensor)

    top_k_neighbors, top_k_scores = find_top_k_nearest_neighbors(
        tower_one_output
        , tower_two_output_dict
        , unique_documents
        , k)

    return [{'score': s, 'document': d} for s, d in zip(top_k_scores, top_k_neighbors)]   