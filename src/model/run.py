from extractor import get_dataset
from train import *

path = 'dataset/games.pgn'
num_games = 5000
device = 'cuda'

X, Y = get_dataset(path,num_games=num_games,device=device)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

encoder = train_encoder(loader,device=device)
print(encoder.parameters)
