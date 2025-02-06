import torch

class TowerOne(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(300,300)
        self.relu=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(300,300)
        self.relu2=torch.nn.ReLU()
        self.fc3=torch.nn.Linear(300,300)
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
class TowerTwo(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=torch.nn.Linear(300,300)
        self.relu=torch.nn.ReLU()
        self.fc2=torch.nn.Linear(300,300)
        self.relu2=torch.nn.ReLU()
        self.fc3=torch.nn.Linear(300,300)   
    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x