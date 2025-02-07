#  run a to get precached document embeddings for new query coming in

import os
import tokeniser
import torch
import tt_models
import numpy as np
import pickle
import generator 
import datasets

script_dir = os.path.dirname(os.path.abspath(__file__))


torch.manual_seed(2)


# Get device
if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device used:", device)

# loading vocab and embeddings
vocab = tokeniser.load_vocab('vocab.json')
id_vectors = np.load('id_vectors.npy', allow_pickle=True).item()


# Process the test dataset
print(f"Loading test parquet...")
ds = datasets.Dataset.from_parquet("ms_marco_validation.parquet")
print(f"Flattening test parquet...")
flat = generator.flatten_dataset(ds)
flat = flat.drop(columns=['query', 'passages.passage_text', 'document_unrelated'])

print(f"Converting to unique dictionary...")
unique_values = flat['document_related'].unique().tolist()

unique_dict = {idx: value for idx, value in enumerate(unique_values)}

# saving to refer to documents for new queries
print(f"Saving unique dictionary with Document text...")
with open('unique_documents_val.pkl', 'wb') as f:
    pickle.dump(unique_dict, f)

print(f"Applying tokens to dictionary...")
length = len(unique_dict)
unique_doc_tokens = {}
for idx, value in unique_dict.items():
    if idx % 10000 == 0:
        print(f"Processing document {idx} of {length}")
    tokens = tokeniser.text_to_ids(value, vocab)
    unique_doc_tokens[idx] = tokens

print(f"Applying embeddings to dictionary...")
unique_doc_embeddings = {}
for idx, value in unique_doc_tokens.items():
    if idx % 10000 == 0:
        print(f"Processing document {idx} of {length}")
    embedding = generator.ids_to_embeddings(value, id_vectors)
    unique_doc_embeddings[idx] = np.mean(np.array(embedding), axis=0)

print(f"Saving dictionary with init mean Document embeddings...")
with open('document_init_embeddings_val.pkl', 'wb') as f:
    pickle.dump(unique_doc_embeddings, f)


t2_model = tt_models.TowerTwo()

print(f"Loading document tower for forward passes...")
with open("epoch2_document_tower.pt", "rb") as f:
    t2_model.load_state_dict(
        torch.load(f, map_location=device, weights_only=True)
    )

t2_model.eval()
t2_model.to(device)

document_tensor = torch.stack([torch.tensor(embedding, dtype=torch.float32) for embedding in unique_doc_embeddings.values()]).to(device)
tower_two_output = t2_model(document_tensor)

tower_two_output_dict = {idx: output.cpu().detach().numpy() for idx, output in enumerate(tower_two_output)}
print("Exported tower_two_output to dictionary.")
with open('document_final_embeddings_val.pkl', 'wb') as f:
    pickle.dump(tower_two_output_dict, f)

# print(len(tower_two_output_dict))
# print(tower_two_output_dict[0])
# print(len(tower_two_output_dict[0]))
