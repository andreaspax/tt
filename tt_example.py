import torch
import wandb
import tqdm


torch.manual_seed(2)

if torch.cuda.is_available():
        device = torch.device("cuda")
elif torch.backends.mps.is_available():
        device = torch.device("mps")
else:
        device = torch.device("cpu")
print("device used:", device)

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
    

sample_size = 131072
queries = torch.randn(sample_size, 300, device=device) # Anchors
positive_documents = queries + torch.randn_like(queries)*0.1  # Similar to docs
negative_documents = torch.randn(sample_size, 300, device=device)   # Random negatives     
dataset = torch.utils.data.TensorDataset(queries, positive_documents, negative_documents)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)

tower_one = TowerOne()
tower_two = TowerTwo()

tower_one.to(device)
tower_two.to(device)


epochs = 1
initial_lr = 0.001
margin = 0.2
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

        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Logging
        wandb.log({"loss": loss.item()})


print("Saving...")
torch.save(tower_one.state_dict(), "./document_tower.pt")
torch.save(tower_two.state_dict(), "./query_tower.pt")
print("Uploading...")
artifact = wandb.Artifact("model-weights", type="model")
artifact.add_file("./document_tower.pt")
artifact.add_file("./query_tower.pt")
wandb.log_artifact(artifact)
print("Done!")
wandb.finish()