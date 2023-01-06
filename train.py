#!/bin/python3
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
import torch 
import torch.optim as optim 
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn 


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.do = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(768, 1048)
        self.fc2 = nn.Linear(1048, 500)
        self.fc3 = nn.Linear(500, 50)
        self.fc4 = nn.Linear(50,1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.do(x)
        x = torch.relu(self.fc2(x))
        x = self.do(x)
        x = torch.relu(self.fc3(x))
        x = self.do(x)
        x = torch.sigmoid(self.fc4(x))
        # x = self.fc4(x)
        return x

if __name__ == '__main__':
    matplotlib.use('tkagg')

    device = 'cuda'
    batch_size = 64

    arr = np.load('dataset/data20k_768.npz')
    X = arr['arr_0']
    Y = arr['arr_1']

    #shuffle
    p = np.random.permutation(len(Y))
    X = X[p]
    Y = Y[p]
    X = torch.tensor(X,dtype=torch.float)
    Y = torch.tensor(Y,dtype=torch.float).view(-1,1)
    del arr
    # Y = Y*2 - 1
    print(f'X - {X.shape} Y - {Y.shape}')

    split = int(X.shape[0] * 0.9)

    train_dataset = TensorDataset(X[:split],Y[:split])
    test_dataset = TensorDataset(X[split:],Y[split:])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    model = Net().to(device)

    epochs = 50

    criterion = nn.BCELoss()
    # optimizer = optim.SGD(model.parameters(),lr=0.1)
    optimizer = optim.Adam(model.parameters(),lr=0.001,betas=(0.9,0.99),eps=1e-08)
    model.train()
    print_interval = len(train_dataset)//(batch_size * 10)
    val_losses = []
    train_losses = []
    test_losses = []

    def test(model):
        model.eval()
        count = 0
        preds = [] 
        ys = []
        t_count = 0
        f_positives = 0
        f_negatives = 0
        running_loss = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred,y)
                count += int(pred.item() > 0.5) == int(y.item()) 
                running_loss += loss.item()
                # if y[0,0] == 1.0 and (pred[0,0] ):
                #     f_negatives += 1
                # if y[0,0] == 0.0 and pred[0,0] == 1:
                #     f_positives += 1
                t_count += 1
        test_loss = running_loss/t_count 
        test_losses.append(test_loss)
        # print(preds[:30])
        # print(ys[:30])
        print(f'accuracy = {count/t_count*100}%, test_loss = {test_loss:.3f}')
        # print(f'fn {f_negatives} fp {f_positives}')

    for epoch in range(epochs):
        t_running_loss = 0
        running_loss = 0
        items = 0
        print(f'-----------------Epoch{epoch+1}-----------------')
        for batch, (x,y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            optimizer.zero_grad()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            t_running_loss += loss.item()
            items += 1
            if batch % print_interval == print_interval - 1:
                print(f'[{epoch+1}, {batch+1}] : {running_loss/print_interval:.3f}')
                running_loss = 0.0
        #if (epoch+1)%5 == 0:
        train_losses.append(t_running_loss/items)
        test(model)
        model.train()

    plt.plot(train_losses, label='train_loss')
    plt.plot(test_losses, label='test_loss')
    plt.legend(loc='lower right')
    plt.show()

    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to disk')
