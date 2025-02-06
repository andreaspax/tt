import os
import tokeniser
import torch
import word2vec
import tt_train

# import embeddings
# from dataset import get_datasets
# from model import DualTowerModel
# from utils import download_weights, get_device

script_dir = os.path.dirname(os.path.abspath(__file__))


torch.manual_seed(2)

if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device used:", device)


t1_model = tt_train.TowerOne()

with open("epoch2_query_tower.pt", "rb") as f:
    t1_model.load_state_dict(
        torch.load(f, map_location=device, weights_only=True)
    )

t1_model.eval()

# # START HERE
query_example = "what can urinalysis detect"
query_embedding = word2vec.text_to_embeddings(query_example, tokeniser.load_vocab('vocab.json'), 'id_vectors.npy').to(device)
print(query_embedding.shape)
print(query_embedding)
# tower_one_output = t1_model.forward(query_embedding)
print(tower_one_output)
# app = FastAPI()
