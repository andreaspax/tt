import torch

torch.manual_seed(2)

class TowerOne(torch.nn.Module):
    def __init__(self):
        super(TowerOne,self).__init__()
        self.fc=torch.nn.Linear(5,3)
    def forward(self, x):
        x = self.fc(x)
        return x
    
class TowerTwo(torch.nn.Module):
    def __init__(self):
        super(TowerOne,self).__init__()
        self.fc = torch.nn.Linear(3,3)
    def forward(self, x):
        x = self.fc(x)
        return x

document = torch.randn(1,5)
query = torch.randn(1,3)

tower_one = TowerOne()
tower_two = TowerTwo()

output_one = tower_one(document)
output_two = tower_two(query)

score = torch.nn.functional.cosine_similarity(output_one, output_two, dim=1)

target = torch.tensor([1.0])
loss = torch.nn.functional.mse_loss(score, target)
loss.backward()
