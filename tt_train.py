import torch
import wandb
import tqdm
import numpy as np



torch.manual_seed(2)

if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device used:", device)

def load_triples(file_path, device):
    # Load numpy array
    triples = np.load(file_path, allow_pickle=True)
    
    # Convert to tensors
    query_tensors = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=device) for t in triples])
    pos_tensors = torch.stack([torch.tensor(t[1], dtype=torch.float32, device=device) for t in triples])
    neg_tensors = torch.stack([torch.tensor(t[2], dtype=torch.float32, device=device) for t in triples])
    
    return query_tensors, pos_tensors, neg_tensors


class TowerOne(torch.nn.Module):
    def __init__(self):
        super(TowerOne,self).__init__()
        self.fc=torch.nn.Linear(300,300)
        self.fc=torch.nn.Linear(300,300)
        self.fc=torch.nn.Linear(300,300)
    def forward(self, x):
        x = self.fc(x)
        return x
    
class TowerTwo(torch.nn.Module):
    def __init__(self):
        super(TowerTwo,self).__init__()
        self.fc=torch.nn.Linear(300,300)
        self.fc=torch.nn.Linear(300,300)
        self.fc=torch.nn.Linear(300,300)    
    def forward(self, x):
        x = self.fc(x)
        return x
    
# Usage in training loop
queries, positive_documents, negative_documents = load_triples('validation_averaged_triples.npy', device)
dataset = torch.utils.data.TensorDataset(queries, positive_documents, negative_documents)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

tower_one = TowerOne()
tower_two = TowerTwo()

tower_one.to(device)
tower_two.to(device)


epochs = 10
initial_lr = 0.001
margin = 0.1
wandb.init(
        project="mlx6-twotowers",
        config={
            "intial_lr": initial_lr,
            "epochs": epochs,
            "margin": margin
        },
    )

optimizer = torch.optim.Adam(
    list(tower_one.parameters()) + list(tower_two.parameters()),
    lr=initial_lr
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
optimizer,
    mode='min',         # Monitor loss
    factor=0.5,         # Reduce LR by 50%
    patience=2,         # Wait 2 epochs with no improvement
    verbose=True,
    min_lr=1e-6
)


for epoch in range(epochs):
    prgs = tqdm.tqdm(dataloader, desc=f"Epoch {epoch+1}", leave=False)
    for qry_batch, doc_pos_batch, doc_neg_batch in prgs:
        optimizer.zero_grad()
        
        # Forward pass
        anchor_emb = tower_one(qry_batch)        # Query tower
        positive_emb = tower_two(doc_pos_batch)  # Document tower
        negative_emb = tower_two(doc_neg_batch) 
        
        # Calculate similarity and loss
        pos_sim = torch.cosine_similarity(anchor_emb, positive_emb, dim=-1)
        neg_sim = torch.cosine_similarity(anchor_emb, negative_emb, dim=-1)
        loss = torch.clamp(margin - (pos_sim - neg_sim), min=0).mean()

        # Update learning rate
        scheduler.step(loss)

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Logging
        wandb.log({
        "loss": loss.item(),
        "pos_sim": pos_sim.mean().item(),
        "neg_sim": neg_sim.mean().item(),
        "learning_rate" : optimizer.param_groups[0]['lr']
        })


    print("Saving...")
    torch.save(tower_one.state_dict(), f"./weights/epoch{epoch+1}_query_tower.pt")
    torch.save(tower_two.state_dict(), f"./weights/epoch{epoch+1}_document_tower.pt")
    print("Uploading...")
    artifact = wandb.Artifact("model-weights", type="model")
    artifact.add_file(f"./weights/epoch{epoch+1}_document_tower.pt")
    artifact.add_file(f"./weights/epoch{epoch+1}_query_tower.pt")
    wandb.log_artifact(artifact)

print("Done!")
wandb.finish()