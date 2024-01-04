#!/bin/python
import torch
from train import Net
import chess
from extractor import encode_board
import numpy as np

model = Net()
model.eval()
model.load_state_dict(torch.load('model.pth'))

b = chess.Board()

def encode(b, moves, color):
    bitboard = np.zeros((len(moves),773))
    for idx, mv in enumerate(moves):
        if mv is not None:
            b.push(mv)
            bitboard[idx] = encode_board(b, color)
            b.pop()
    return bitboard


def analyze(b, moves, turn):
    with torch.no_grad():
        encoded_tensor = torch.tensor(encode(b, moves, turn%2),dtype=torch.float)
        pred = model(encoded_tensor)
        #pred = pred[:,1] - pred[:,0]
        idx = torch.argmax(-pred)
        return moves[idx]

turn = 0

while(True):
    if turn%2 == 0:
        #white's turn
        print('white to play')
        mv = input('enter your move ')
        try:
            b.push_san(mv)
        except ValueError:
            mv = input('enter your move again ')
            b.push_san(mv)
    else:
        print('black to play')
        mv = analyze(b,list(b.legal_moves),turn)
        b.push(mv)
        print(mv)
    print(b)
    turn += 1
