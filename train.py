import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from matplotlib import pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(773,600),
            nn.ReLU(),
            nn.Linear(600,400),
            nn.ReLU(),
            nn.Linear(400,200),
            nn.ReLU(),
            nn.Linear(200,100)
        )

        self.decoder = nn.Sequential(
            nn.Linear(100,200),
            nn.ReLU(),
            nn.Linear(200,400),
            nn.ReLU(),
            nn.Linear(400,600),
            nn.ReLU(),
            nn.Linear(600,773),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def train_encoder(dataloader, device='cpu'):
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)

    epochs = 10
    losses = []

    print("Training autoencoder")

    running_loss = 0
    for epoch in range(epochs):
        for batch, (x, _) in enumerate(dataloader):
            batch_loss = 0
            out = model(x)
            loss = criterion(out,x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if batch % 500 == 499:
                print(f'[{epoch + 1}, {batch + 1}] loss: {running_loss / 500:.3f}')
                running_loss = 0.0
        losses.append(running_loss)
    
    print('finished training autoencoder ')
    
    return model.encoder