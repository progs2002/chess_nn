#!/bin/python3
import numpy as np 
import matplotlib.pyplot as plt 
import torch 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn 

device = 'cuda'
batch_size = 128

arr = np.load('data20k.npz')
X = arr['arr_0']
Y = arr['arr_1']

#shuffle
p = np.random.permutation(len(Y))
X = X[p]
Y = Y[p]
Y_t = np.ones((Y.shape[0],2))
Y_t[:,0] = Y
Y_t[:,1] = np.where((Y==0)|(Y==1), Y^1, Y)

X = torch.tensor(X,dtype=torch.float,device=device)
Y = torch.tensor(Y_t,dtype=torch.float,device=device)
# Y = Y*2 - 1
print(f'X - {X.shape} Y - {Y.shape}')

split = int(X.shape[0] * 0.9)

train_dataset = TensorDataset(X[:split],Y[:split])
test_dataset = TensorDataset(X[split:],Y[split:])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(773, 600)
        self.fc2 = nn.Linear(600, 300)
        self.fc3 = nn.Linear(300, 100)
        self.fc4 = nn.Linear(100,50)
        self.fc5 = nn.Linear(50,2)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = Net().to(device)

epochs = 200

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.1)

print_interval = len(train_dataset)//(batch_size * 10)

for epoch in range(epochs):
    running_loss = 0
    print(f'-----------------Epoch{epoch+1}-----------------')
    for batch, (x,y) in enumerate(train_loader):
        pred = model(x)
        optimizer.zero_grad()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if batch % print_interval == print_interval - 1:
            print(f'[{epoch+1}, {batch+1}] : {running_loss/print_interval:.3f}')
            running_loss = 0.0

count = 0
preds = [] 
ys = []
t_count = 0
f_positives = 0
f_negatives = 0
with torch.no_grad():
    for x, y in test_loader:
        pred = model(x)
        count += torch.sum(torch.argmax(pred,1) == torch.argmax(y,1))
        # if y[0,0] == 1.0 and (pred[0,0] ):
        #     f_negatives += 1
        # if y[0,0] == 0.0 and pred[0,0] == 1:
        #     f_positives += 1
        t_count += 1

# print(preds[:30])
# print(ys[:30])
print(f'accuracy = {count/t_count*100}%')
# print(f'fn {f_negatives} fp {f_positives}')

torch.save(model.state_dict(), 'model.pth')
print('Model saved to disk')

